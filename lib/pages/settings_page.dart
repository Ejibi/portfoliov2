import 'package:flutter/material.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  bool _notificationsEnabled = true;
  bool _autoAnalyzeEnabled = false;
  double _confidenceThreshold = 0.75;
  
  late List<Map<String, dynamic>> _sections;

  @override
  void initState() {
    super.initState();
    _sections = [
      {'id': 'general', 'title': 'General Settings', 'icon': Icons.tune},
      {'id': 'ai', 'title': 'AI Classification Model', 'icon': Icons.psychology},
      {'id': 'display', 'title': 'Display & Interface', 'icon': Icons.monitor},
      {'id': 'data', 'title': 'Data & Local Storage', 'icon': Icons.storage},
    ];
  }

  Widget _buildSectionContent(String id) {
    switch (id) {
      case 'general':
        return Column(
          children: [
            _SettingRow(label: 'System Language', trailing: const Text('English (US) >')),
            _SettingRow(
              label: 'Notifications',
              trailing: Switch(
                value: _notificationsEnabled,
                onChanged: (val) => setState(() => _notificationsEnabled = val),
                activeThumbColor: Colors.cyan,
              ),
            ),
          ],
        );
      case 'ai':
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Confidence Threshold: ${(_confidenceThreshold * 100).toInt()}%', style: const TextStyle(fontSize: 14)),
            Slider(
              value: _confidenceThreshold,
              onChanged: (val) => setState(() => _confidenceThreshold = val),
              activeColor: Colors.cyan,
            ),
            const _SettingRow(label: 'Max Classes Output', trailing: Text('8 Classes >')),
          ],
        );
      case 'display':
        return Column(
          children: [
            const _SettingRow(label: 'Default Canvas Zoom', trailing: Text('Fit-to-Screen (Auto) >')),
            const _SettingRow(label: 'Result Density', trailing: Text('Comfortable >')),
            _SettingRow(
              label: 'Auto-Analyze',
              trailing: Switch(
                value: _autoAnalyzeEnabled,
                onChanged: (val) => setState(() => _autoAnalyzeEnabled = val),
                activeThumbColor: Colors.cyan,
              ),
            ),
          ],
        );
      case 'data':
        return Column(
          children: [
            const _SettingRow(label: 'Local Image Cache', trailing: Text('254.8 MB')),
            Align(
              alignment: Alignment.centerRight,
              child: TextButton(onPressed: () {}, child: const Text('Clear Cache', style: TextStyle(color: Colors.white70))),
            ),
            const _SettingRow(label: 'Export Annotation Format', trailing: Text('COCO JSON >')),
          ],
        );
      default:
        return const SizedBox.shrink();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(48.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Settings', style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold)),
                  Text('CONFIGURE SYSTEM PREFERENCES & INTERFACES', 
                    style: TextStyle(color: Colors.grey, fontSize: 10, letterSpacing: 1.5)),
                ],
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.green),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: const Text('CHANGES SAVED', style: TextStyle(color: Colors.green, fontSize: 10, fontWeight: FontWeight.bold)),
              ),
            ],
          ),
          const SizedBox(height: 48),
          Expanded(
            child: ListView(
              children: _sections.map((section) {
                return Padding(
                  key: ValueKey(section['id']),
                  padding: const EdgeInsets.only(bottom: 24.0),
                  child: _SettingsSection(
                    icon: section['icon'],
                    title: section['title'],
                    child: _buildSectionContent(section['id']),
                  ),
                );
              }).toList(),
            ),
          ),
        ],
      ),
    );
  }
}

class _SettingsSection extends StatelessWidget {
  final IconData icon;
  final String title;
  final Widget child;

  const _SettingsSection({required this.icon, required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: const Color(0xFF16161A),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: Colors.cyan, size: 20),
              const SizedBox(width: 12),
              Text(title, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            ],
          ),
          const SizedBox(height: 24),
          child,
        ],
      ),
    );
  }
}

class _SettingRow extends StatelessWidget {
  final String label;
  final Widget trailing;

  const _SettingRow({required this.label, required this.trailing});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: Colors.white70)),
          trailing,
        ],
      ),
    );
  }
}
