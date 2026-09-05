import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';
import 'package:firebase_auth/firebase_auth.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final TransformationController _zoomController = TransformationController();
  double _currentScale = 1.0;
  Offset? _focusPoint;
  final double _boxSize = 120.0;

  // Real-time State
  Uint8List? _uploadedImageBytes;
  String? _selectedClassFilter; // Null = base image, 'ALL' = composite, or class name e.g. 'crack'
  Map<String, Uint8List> _maskImageMap = {}; // Maps class name -> mask overlay bytes
  bool _isLoading = false;
  String _inferenceTime = '0.0ms';

  // Dynamic Detections driven by predict.py damage_summary
  List<Map<String, dynamic>> _detections = [];

  void _handleTap(TapDownDetails details) {
    setState(() => _focusPoint = details.localPosition);
  }

  void _updateZoom(double step, Size containerSize) {
    setState(() {
      _currentScale = (_currentScale + step).clamp(1.0, 5.0);
      if (_focusPoint != null && _currentScale > 1.0) {
        final tx = (containerSize.width / 2) - (_focusPoint!.dx * _currentScale);
        final ty = (containerSize.height / 2) - (_focusPoint!.dy * _currentScale);
        _zoomController.value = Matrix4.identity()
          ..translateByDouble(tx, ty, 0.0, 1.0)
          ..scaleByDouble(_currentScale, _currentScale, 1.0, 1.0);
      } else {
        _zoomController.value = Matrix4.identity()..scaleByDouble(_currentScale, _currentScale, 1.0, 1.0);
      }
    });
  }

  Future<void> _pickAndPredictImage() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      withData: true,
    );

    if (result == null || result.files.single.bytes == null) return;

    final imageBytes = result.files.single.bytes!;
    setState(() {
      _uploadedImageBytes = imageBytes;
      _isLoading = true;
      _focusPoint = null;
      _selectedClassFilter = 'ALL'; // Default to full overlay after upload
      _maskImageMap.clear();
    });

    final stopwatch = Stopwatch()..start();

    try {
      final request = http.MultipartRequest('POST', Uri.parse('/api/predict'));

      final user = FirebaseAuth.instance.currentUser;
      if (user != null) {
        final token = await user.getIdToken();
        request.headers['Authorization'] = 'Bearer $token';
      }

      request.files.add(http.MultipartFile.fromBytes('file', imageBytes, filename: result.files.single.name));


      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);
      stopwatch.stop();

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);

        // 1. Map predict.py damage_summary per_class dict directly to _detections
        final Map<String, dynamic> perClass = data['summary']['per_class'] ?? {};
        final List<Map<String, dynamic>> parsedDetections = [];

        int idCounter = 1;
        perClass.forEach((className, stats) {
          final double pct = (stats['damage_pct'] as num).toDouble();
          if (pct > 0.0) { // Only show active damage classes
            parsedDetections.add({
              'id': idCounter.toString().padLeft(2, '0'),
              'class': className,
              'damage_pct': pct,
              'severity': stats['severity'],
            });
            idCounter++;
          }
        });

        // 2. Map Base64 mask images returned from Modal (predict.py outputs)
        final Map<String, dynamic> rawMasks = data['masks_base64'] ?? {};
        final Map<String, Uint8List> decodedMasks = {};
        rawMasks.forEach((key, b64) {
          decodedMasks[key] = base64Decode(b64);
        });

        setState(() {
          _inferenceTime = '${stopwatch.elapsedMilliseconds}ms';
          _detections = parsedDetections;
          _maskImageMap = decodedMasks; // Stores 'ALL', 'crack', 'dent', etc.
        });
      }
    } catch (e) {
      debugPrint('Prediction error: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  void _showUploadPopup() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF16161A),
        title: const Text('Load Image Source'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.computer, color: Colors.cyan),
              title: const Text('Local System'),
              onTap: () {
                Navigator.pop(context);
                _pickAndPredictImage();
              },
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final Uint8List? activeMaskBytes = _selectedClassFilter != null ? _maskImageMap[_selectedClassFilter] : null;

    return LayoutBuilder(
      builder: (context, constraints) {
        final bool isMobile = constraints.maxWidth < 1000;

        return Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('ai-image-classifier', style: TextStyle(color: Colors.grey, fontSize: 18)),
              const SizedBox(height: 24),
              Expanded(
                child: isMobile
                    ? ListView(
                        children: [
                          _buildImageSection(activeMaskBytes),
                          const SizedBox(height: 24),
                          _buildAnalysisPanel(),
                        ],
                      )
                    : Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Expanded(flex: 2, child: SingleChildScrollView(child: _buildImageSection(activeMaskBytes))),
                          const SizedBox(width: 24),
                          Expanded(child: SingleChildScrollView(child: _buildAnalysisPanel())),
                        ],
                      ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildImageSection(Uint8List? activeMaskBytes) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: Colors.white10),
          ),
          clipBehavior: Clip.antiAlias,
          child: AspectRatio(
            aspectRatio: 16 / 9,
            child: LayoutBuilder(
              builder: (context, constraints) {
                final Size size = Size(constraints.maxWidth, constraints.maxHeight);
                return Stack(
                  children: [
                    InteractiveViewer(
                      transformationController: _zoomController,
                      minScale: 1.0,
                      maxScale: 5.0,
                      boundaryMargin: const EdgeInsets.all(double.infinity),
                      onInteractionUpdate: (_) {
                        setState(() => _currentScale = _zoomController.value.getMaxScaleOnAxis());
                      },
                      child: GestureDetector(
                        onTapDown: _handleTap,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            _uploadedImageBytes != null
                                ? Image.memory(_uploadedImageBytes!, fit: BoxFit.cover)
                                : Image.network('https://picsum.photos/id/10/800/450', fit: BoxFit.cover),
                            if (activeMaskBytes != null)
                              Image.memory(activeMaskBytes, fit: BoxFit.cover),
                          ],
                        ),
                      ),
                    ),
                    if (_isLoading)
                      Container(
                        color: Colors.black54,
                        child: const Center(child: CircularProgressIndicator(color: Colors.cyan)),
                      ),
                    if (_focusPoint != null)
                      Builder(
                        builder: (context) {
                          final screenPos = MatrixUtils.transformPoint(_zoomController.value, _focusPoint!);
                          return Positioned(
                            left: screenPos.dx - (_boxSize / 2),
                            top: screenPos.dy - (_boxSize / 2),
                            child: IgnorePointer(
                              child: Container(
                                width: _boxSize,
                                height: _boxSize,
                                decoration: BoxDecoration(
                                  border: Border.all(color: Colors.cyan, width: 2),
                                  boxShadow: [BoxShadow(color: Colors.cyan.withValues(alpha: 0.1), blurRadius: 8, spreadRadius: 1)],
                                ),
                              ),
                            ),
                          );
                        },
                      ),
                    Positioned(
                      top: 16,
                      right: 16,
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 4),
                        decoration: BoxDecoration(color: Colors.black54, borderRadius: BorderRadius.circular(8)),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            IconButton(
                              icon: const Icon(Icons.remove, size: 16), 
                              onPressed: () => _updateZoom(-0.5, size)
                            ),
                            Padding(
                              padding: const EdgeInsets.symmetric(horizontal: 8.0),
                              child: Text('${(_currentScale * 100).toInt()}%', style: const TextStyle(fontSize: 12)),
                            ),
                            IconButton(
                              icon: const Icon(Icons.add, size: 16), 
                              onPressed: () => _updateZoom(0.5, size)
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                );
              },
            ),
          ),
        ),
        const SizedBox(height: 16),
        const Text('DETECTION_CHIP_FEED (CLICK TO FILTER MASK)', style: TextStyle(color: Colors.grey, fontSize: 10, letterSpacing: 1.2)),
        const SizedBox(height: 12),
        SizedBox(
          height: 32,
          child: ListView(
            scrollDirection: Axis.horizontal,
            children: [
              GestureDetector(
                onTap: () => setState(() => _selectedClassFilter = 'ALL'),
                child: Container(
                  margin: const EdgeInsets.only(right: 8),
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: _selectedClassFilter == 'ALL' ? Colors.cyan : Colors.white10,
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Text('ALL OVERLAYS', style: TextStyle(color: _selectedClassFilter == 'ALL' ? Colors.black : Colors.white, fontSize: 12, fontWeight: FontWeight.bold)),
                ),
              ),
              ..._detections.map((det) {
                final String className = det['class'];
                final bool isSelected = _selectedClassFilter == className;
                return GestureDetector(
                  onTap: () => setState(() => _selectedClassFilter = isSelected ? null : className),
                  child: Container(
                    margin: const EdgeInsets.only(right: 8),
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: isSelected ? Colors.cyan : Colors.cyan.withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: Colors.cyan),
                    ),
                    child: Text(
                      '${det['class']} ${det['damage_pct'].toStringAsFixed(1)}%',
                      style: TextStyle(color: isSelected ? Colors.black : Colors.cyan, fontSize: 12, fontWeight: isSelected ? FontWeight.bold : FontWeight.normal),
                    ),
                  ),
                );
              }),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildAnalysisPanel() {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(color: const Color(0xFF16161A), borderRadius: BorderRadius.circular(12)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Wrap(
            spacing: 12,
            runSpacing: 12,
            alignment: WrapAlignment.spaceBetween,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(border: Border.all(color: _isLoading ? Colors.amber : Colors.green), borderRadius: BorderRadius.circular(4)),
                child: Text(_isLoading ? 'PROCESSING...' : 'ANALYSIS COMPLETE', style: TextStyle(color: _isLoading ? Colors.amber : Colors.green, fontSize: 10, fontWeight: FontWeight.bold)),
              ),
              ElevatedButton.icon(
                onPressed: _showUploadPopup,
                icon: const Icon(Icons.upload, size: 16, color: Colors.black),
                label: const Text('Load Image', style: TextStyle(color: Colors.black, fontSize: 12)),
                style: ElevatedButton.styleFrom(backgroundColor: Colors.cyan),
              ),
            ],
          ),
          const SizedBox(height: 24),
          const Text('INFERENCE TIME', style: TextStyle(color: Colors.grey, fontSize: 10)),
          Text(_inferenceTime, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
          const SizedBox(height: 32),
          const Text('Detected Classes (Damage %)', style: TextStyle(fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          ..._detections.map((det) {
            final bool isSelected = _selectedClassFilter == det['class'];
            return Padding(
              padding: const EdgeInsets.only(bottom: 20.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Flexible(
                        child: Text(
                          '[${det['id']}] ${det['class']} (${det['severity']})',
                          style: TextStyle(
                            color: isSelected ? Colors.cyan : Colors.white,
                            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                          ),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text('${det['damage_pct'].toStringAsFixed(2)}%', style: const TextStyle(color: Colors.cyan, fontSize: 12)),
                    ],
                  ),
                  const SizedBox(height: 8),
                  LinearProgressIndicator(
                    value: (det['damage_pct'] as double) / 100,
                    backgroundColor: Colors.white12,
                    valueColor: AlwaysStoppedAnimation<Color>(isSelected ? Colors.cyan : Colors.cyan.withValues(alpha: 0.4)),
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }
}
