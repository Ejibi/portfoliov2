import 'package:flutter/material.dart';
import 'pages/home_page.dart';
import 'pages/about_page.dart';
import 'pages/settings_page.dart';
import 'pages/account_page.dart';
import 'pages/vehicle_events_page.dart';

class RootNavigation extends StatefulWidget {
  const RootNavigation({super.key});

  @override
  State<RootNavigation> createState() => _RootNavigationState();
}

class _RootNavigationState extends State<RootNavigation> {
  int _selectedIndex = 0;
  bool _isCollapsed = false;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        // Auto-collapse if width is small (e.g., tablet/mobile)
        final bool autoCollapse = constraints.maxWidth < 900;
        final bool effectiveCollapsed = _isCollapsed || autoCollapse;

        return Scaffold(
          backgroundColor: const Color(0xFF0D0D0F),
          body: Row(
            children: [
              // Sidebar
              Sidebar(
                selectedIndex: _selectedIndex,
                isCollapsed: effectiveCollapsed,
                onItemSelected: (index) => setState(() => _selectedIndex = index),
                onToggleCollapse: () => setState(() => _isCollapsed = !_isCollapsed),
              ),
              // Main Content
              Expanded(
                child: Container(
                  decoration: BoxDecoration(
                    color: const Color(0xFF121214),
                    borderRadius: constraints.maxWidth < 600 
                        ? BorderRadius.zero 
                        : const BorderRadius.only(topLeft: Radius.circular(32)),
                  ),
                  clipBehavior: Clip.antiAlias,
                  child: IndexedStack(
                    index: _selectedIndex,
                    children: const [
                      HomePage(),
                      AboutPage(),
                      SettingsPage(),
                      AccountPage(),
                      VehicleEventsPage(),
                    ],
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}
class Sidebar extends StatelessWidget {
  final int selectedIndex;
  final bool isCollapsed;
  final Function(int) onItemSelected;
  final VoidCallback onToggleCollapse;

  const Sidebar({
    super.key,
    required this.selectedIndex,
    required this.isCollapsed,
    required this.onItemSelected,
    required this.onToggleCollapse,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeInOut,
      width: isCollapsed ? 80 : 240,
      color: const Color(0xFF0D0D0F),
      clipBehavior: Clip.antiAlias,
      child: AnimatedPadding(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
        padding: EdgeInsets.symmetric(horizontal: isCollapsed ? 8 : 12, vertical: 24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            // App Logo/Name
            Row(
              mainAxisAlignment: isCollapsed ? MainAxisAlignment.center : MainAxisAlignment.start,
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.cyan,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Icon(Icons.auto_awesome, color: Colors.black, size: 20),
                ),
                ClipRect(
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 300),
                    curve: Curves.easeInOut,
                    width: isCollapsed ? 0 : 150,
                    child: SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      physics: const NeverScrollableScrollPhysics(),
                      child: Row(
                        children: [
                          const SizedBox(width: 12),
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisSize: MainAxisSize.min,
                            children: const [
                              Text(
                                'Portfolio_v2',
                                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                              ),
                              Text(
                                'CLASSIFY v1.0',
                                style: TextStyle(fontSize: 10, color: Colors.grey),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),

            // Pages Navigation List
            Expanded(
              child: SingleChildScrollView(
                child: Column(
                  children: [
                    _SidebarItem(
                      icon: Icons.home_outlined,
                      label: 'Home',
                      isSelected: selectedIndex == 0,
                      onTap: () => onItemSelected(0),
                      isCollapsed: isCollapsed,
                    ),
                    _SidebarItem(
                      icon: Icons.info_outline,
                      label: 'About',
                      isSelected: selectedIndex == 1,
                      onTap: () => onItemSelected(1),
                      isCollapsed: isCollapsed,
                    ),
                    _SidebarItem(
                      icon: Icons.settings_outlined,
                      label: 'Settings',
                      isSelected: selectedIndex == 2,
                      onTap: () => onItemSelected(2),
                      isCollapsed: isCollapsed,
                    ),
                    _SidebarItem(
                      icon: Icons.person_outline,
                      label: 'Account',
                      isSelected: selectedIndex == 3,
                      onTap: () => onItemSelected(3),
                      isCollapsed: isCollapsed,
                    ),
                    _SidebarItem(
                      icon: Icons.calendar_month,
                      label: 'Vehicle Events',
                      isSelected: selectedIndex == 4,
                      onTap: () => onItemSelected(4),
                      isCollapsed: isCollapsed,
                    ),
                  ],
                ),
              ),
            ),

            // Toggle Button at the bottom
            const SizedBox(height: 16),
            AnimatedAlign(
              duration: const Duration(milliseconds: 300),
              curve: Curves.easeInOut,
              alignment: isCollapsed ? Alignment.center : Alignment.centerRight,
              child: IconButton(
                icon: Icon(
                  isCollapsed ? Icons.chevron_right : Icons.chevron_left,
                  color: Colors.grey,
                ),
                onPressed: onToggleCollapse,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SidebarItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool isSelected;
  final VoidCallback onTap;
  final bool isCollapsed;

  const _SidebarItem({
    required this.icon,
    required this.label,
    required this.isSelected,
    required this.onTap,
    required this.isCollapsed,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeInOut,
          padding: EdgeInsets.symmetric(horizontal: isCollapsed ? 0 : 16, vertical: 12),
          decoration: BoxDecoration(
            color: isSelected ? Colors.cyan.withValues(alpha: 0.1) : Colors.transparent,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: isSelected ? Colors.cyan.withValues(alpha: 0.3) : Colors.transparent),
          ),
          child: Row(
            mainAxisAlignment: isCollapsed ? MainAxisAlignment.center : MainAxisAlignment.start,
            children: [
              Icon(icon, color: isSelected ? Colors.cyan : Colors.grey, size: 20),
              ClipRect(
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 300),
                  curve: Curves.easeInOut,
                  width: isCollapsed ? 0 : 130,
                  child: SingleChildScrollView(
                    scrollDirection: Axis.horizontal,
                    physics: const NeverScrollableScrollPhysics(),
                    child: Row(
                      children: [
                        const SizedBox(width: 16),
                        Text(
                          label,
                          style: TextStyle(
                            color: isSelected ? Colors.cyan : Colors.grey,
                            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}