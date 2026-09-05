import 'package:flutter/material.dart';

class VehicleEventsPage extends StatefulWidget {
  const VehicleEventsPage({super.key});

  @override
  State<VehicleEventsPage> createState() => _VehicleEventsPageState();
}

class _VehicleEventsPageState extends State<VehicleEventsPage> {
  DateTime _currentWeekStart = DateTime(DateTime.now().year, DateTime.now().month, DateTime.now().day).subtract(Duration(days: DateTime.now().weekday - 1));
  DateTime? _selectionDate;
  int? _startHour;
  int? _endHour;
  
  // Storage for events: Map<DateKey, List<Event>>
  final Map<String, List<Map<String, String>>> _events = {};

  String _getDateKey(DateTime date) {
    return "${date.year}-${date.month}-${date.day}";
  }

  void _pickDate() async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: _selectionDate ?? DateTime.now(),
      firstDate: DateTime(2020),
      lastDate: DateTime(2030),
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: const ColorScheme.dark(
              primary: Colors.cyan,
              onPrimary: Colors.black,
              surface: const Color(0xFF16161A),
              onSurface: Colors.white,
            ),
          ),
          child: child!,
        );
      },
    );
    if (picked != null) {
      setState(() {
        _selectionDate = picked;
        _currentWeekStart = picked.subtract(Duration(days: picked.weekday - 1));
        _startHour = null;
        _endHour = null;
      });
    }
  }

  void _addEvent() {
    if (_selectionDate == null || _startHour == null) return;
    
    final int endHour = _endHour ?? _startHour!;
    final titleController = TextEditingController();
    final descController = TextEditingController();

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF16161A),
        title: const Text('Add Event', style: TextStyle(color: Colors.white)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: titleController,
              decoration: const InputDecoration(
                labelText: 'Title',
                labelStyle: TextStyle(color: Colors.cyan),
              ),
              style: const TextStyle(color: Colors.white),
            ),
            TextField(
              controller: descController,
              decoration: const InputDecoration(
                labelText: 'Description',
                labelStyle: TextStyle(color: Colors.cyan),
              ),
              style: const TextStyle(color: Colors.white),
            ),
            const SizedBox(height: 16),
            Text(
              'Time: $_startHour:00 - ${endHour + 1}:00',
              style: const TextStyle(color: Colors.grey),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel', style: TextStyle(color: Colors.grey)),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: Colors.cyan),
            onPressed: () {
              setState(() {
                final key = _getDateKey(_selectionDate!);
                _events.putIfAbsent(key, () => []);
                _events[key]!.add({
                  'title': titleController.text,
                  'description': descController.text,
                  'range': '$_startHour:00 - ${endHour + 1}:00',
                  'start': _startHour.toString(),
                  'end': (endHour + 1).toString(),
                });
                _startHour = null;
                _endHour = null;
              });
              Navigator.pop(context);
            },
            child: const Text('Save', style: TextStyle(color: Colors.black)),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          children: [
            // Header spans across the top
            _buildHeader(),
            const SizedBox(height: 24),
            
            Expanded(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Main Content: Week View Grid (Left)
                  Expanded(
                    flex: 3,
                    child: Column(
                      children: [
                        Expanded(
                          child: Container(
                            decoration: BoxDecoration(
                              color: const Color(0xFF16161A),
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(color: Colors.white10),
                            ),
                            child: _buildWeekGrid(),
                          ),
                        ),
                        const SizedBox(height: 16),
                        // Bottom Action stays associated with the grid
                        _buildBottomAction(),
                      ],
                    ),
                  ),
                  
                  const SizedBox(width: 24),
                  
                  // Right Sidebar: Tracked Events
                  Expanded(
                    flex: 1,
                    child: Container(
                      decoration: BoxDecoration(
                        color: const Color(0xFF16161A),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: Colors.white10),
                      ),
                      child: _buildAllEventsSidebar(),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    final displayDate = _selectionDate ?? DateTime.now();
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          '${displayDate.day.toString().padLeft(2, '0')}/${displayDate.month.toString().padLeft(2, '0')}/${displayDate.year}',
          style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white),
        ),
        IconButton(
          icon: const Icon(Icons.calendar_month, color: Colors.cyan, size: 30),
          onPressed: _pickDate,
        ),
      ],
    );
  }

  Widget _buildAllEventsSidebar() {
    List<Map<String, dynamic>> allEvents = [];
    _events.forEach((dateKey, list) {
      for (int i = 0; i < list.length; i++) {
        allEvents.add({
          'dateKey': dateKey,
          'index': i,
          ...list[i],
        });
      }
    });

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Padding(
          padding: EdgeInsets.all(24.0),
          child: Text(
            'TRACKED EVENTS',
            style: TextStyle(color: Colors.cyan, fontWeight: FontWeight.bold, letterSpacing: 1.5, fontSize: 12),
          ),
        ),
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            itemCount: allEvents.length,
            itemBuilder: (context, index) {
              final event = allEvents[index];
              return Container(
                margin: const EdgeInsets.only(bottom: 12),
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.05),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  children: [
                    Container(width: 3, height: 40, color: Colors.cyan),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(event['title']!, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                          Text('${event['dateKey']} • ${event['range']}', style: TextStyle(color: Colors.white.withValues(alpha: 0.5), fontSize: 11)),
                          Text(event['description']!, style: TextStyle(color: Colors.white.withValues(alpha: 0.3), fontSize: 11), overflow: TextOverflow.ellipsis),
                        ],
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.close, color: Colors.grey, size: 16),
                      onPressed: () {
                        setState(() {
                          _events[event['dateKey']]!.removeAt(event['index']);
                          if (_events[event['dateKey']]!.isEmpty) _events.remove(event['dateKey']);
                        });
                      },
                    ),
                  ],
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _buildWeekGrid() {
    final weekDays = List.generate(7, (index) => _currentWeekStart.add(Duration(days: index)));

    return Column(
      children: [
        // Day Headers Row
        Container(
          height: 60,
          decoration: const BoxDecoration(
            border: Border(bottom: BorderSide(color: Colors.white10)),
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(width: 60), // Space for hour labels
              const VerticalDivider(color: Colors.white10, width: 1),
              ...weekDays.map((date) => Expanded(
                child: GestureDetector(
                  behavior: HitTestBehavior.opaque,
                  onTap: () => setState(() { 
                    _selectionDate = date; 
                    _startHour = null; 
                    _endHour = null; 
                  }),
                  child: Container(
                    decoration: BoxDecoration(
                      color: (_selectionDate != null && _getDateKey(date) == _getDateKey(_selectionDate!)) 
                        ? Colors.cyan.withOpacity(0.1) 
                        : Colors.transparent,
                      border: Border(right: BorderSide(color: Colors.white.withOpacity(0.05))),
                    ),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'][date.weekday - 1], 
                          style: const TextStyle(color: Colors.grey, fontSize: 10, fontWeight: FontWeight.bold)),
                        Text(date.day.toString(), 
                          style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
                      ],
                    ),
                  ),
                ),
              )),
            ],
          ),
        ),
        
        // Scrollable Grid
        Expanded(
          child: ListView.builder(
            itemCount: 24,
            itemBuilder: (context, hour) {
              return Container(
                height: 60,
                decoration: BoxDecoration(
                  border: Border(bottom: BorderSide(color: Colors.white.withOpacity(0.05))),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // Hour Label
                    Container(
                      width: 60,
                      alignment: Alignment.center,
                      child: Text('${hour.toString().padLeft(2, '0')}:00', style: const TextStyle(color: Colors.grey, fontSize: 10)),
                    ),
                    const VerticalDivider(color: Colors.white10, width: 1),
                    // Day Columns
                    ...weekDays.map((date) {
                      final dateKey = _getDateKey(date);
                      bool isSelectedDay = _selectionDate != null && _getDateKey(_selectionDate!) == dateKey;
                      bool isInRange = false;
                      if (isSelectedDay) {
                        if (_startHour != null && _endHour != null) {
                          isInRange = hour >= _startHour! && hour <= _endHour!;
                        } else if (_startHour == hour) {
                          isInRange = true;
                        }
                      }

                      return Expanded(
                        child: GestureDetector(
                          behavior: HitTestBehavior.opaque,
                          onTap: () => _onSlotTap(date, hour),
                          child: Container(
                            decoration: BoxDecoration(
                              color: isInRange ? Colors.cyan.withValues(alpha: 0.3) : Colors.transparent,
                              border: Border(right: BorderSide(color: Colors.white.withValues(alpha: 0.02))),
                            ),
                            child: _buildSlotEvents(dateKey, hour),
                          ),
                        ),
                      );
                    }),
                  ],
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  void _onSlotTap(DateTime date, int hour) {
    setState(() {
      final String dateKey = _getDateKey(date);
      final String? selectedKey = _selectionDate != null ? _getDateKey(_selectionDate!) : null;

      if (_selectionDate == null || selectedKey != dateKey) {
        // New day selected
        _selectionDate = date;
        _startHour = hour;
        _endHour = null;
      } else {
        // Same day: handle range selection
        if (_startHour == null || (_startHour != null && _endHour != null)) {
          _startHour = hour;
          _endHour = null;
        } else {
          if (hour < _startHour!) {
            _endHour = _startHour;
            _startHour = hour;
          } else if (hour > _startHour!) {
            _endHour = hour;
          } else {
            // Tapped same hour: toggle off or keep as single hour
            _endHour = hour;
          }
        }
      }
    });
  }

  Widget _buildSlotEvents(String dateKey, int hour) {
    final dayEvents = _events[dateKey] ?? [];
    final eventsInHour = dayEvents.where((e) {
      if (e.containsKey('start') && e.containsKey('end')) {
        int start = int.parse(e['start']!);
        int end = int.parse(e['end']!);
        return hour >= start && hour < end;
      }
      return e['range']!.startsWith('${hour}:00');
    }).toList();

    if (eventsInHour.isEmpty) return const SizedBox.shrink();

    return Column(
      children: eventsInHour.map((e) => Container(
        margin: const EdgeInsets.all(2),
        padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
        decoration: BoxDecoration(
          color: Colors.cyan,
          borderRadius: BorderRadius.circular(2),
        ),
        child: Text(
          e['title']!,
          style: const TextStyle(color: Colors.black, fontSize: 8, fontWeight: FontWeight.bold),
          overflow: TextOverflow.ellipsis,
        ),
      )).toList(),
    );
  }

  Widget _buildBottomAction() {
    bool canCreate = _selectionDate != null && _startHour != null;
    return SizedBox(
      width: double.infinity,
      height: 50,
      child: ElevatedButton(
        onPressed: canCreate ? _addEvent : null,
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.cyan,
          disabledBackgroundColor: Colors.grey.withOpacity(0.1),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        ),
        child: Text(
          'CREATE EVENT',
          style: TextStyle(
            color: canCreate ? Colors.black : Colors.grey,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
    );
  }
}
