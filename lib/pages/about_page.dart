import 'package:flutter/material.dart';

class AboutPage extends StatelessWidget {
  const AboutPage({super.key});

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(64.0),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 800),
          child: Column(
            children: [
              const Text('About', style: TextStyle(fontSize: 48, fontWeight: FontWeight.w200)),
              const SizedBox(height: 48),
              const Text(
                'The core essence of website is to combine my portfolio projects in a way that meshes them together into a final useful product.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 18, color: Colors.white70),
              ),
              const SizedBox(height: 24),
              const Text(
                'I want what I bring to the world to serve a purpose and be used above all else.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 18, color: Colors.white70),
              ),
              const SizedBox(height: 24),
              const Text(
                'The sum becoming whole is greater than the essence of each individual component examined in isolation.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 18, color: Colors.white70),
              ),
              const SizedBox(height: 64),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text('Individual Components', style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.white10,
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: const Text('10 Items', style: TextStyle(fontSize: 12, color: Colors.grey)),
                  ),
                ],
              ),
              const SizedBox(height: 24),
              const _ComponentItem(icon: Icons.layers, title: 'Project Canvas', subtitle: 'Main page with AI model and damage detection'),
              const _ComponentItem(icon: Icons.web_asset, title: 'Sidebar Navigation', subtitle: 'Compact navigation with active state and subtle hover affordances.'),
              const _ComponentItem(icon: Icons.description_outlined, title: 'About Section', subtitle: 'Info page talking about the project and its purpose'),
              const _ComponentItem(icon: Icons.calendar_month, title: 'Vehicle Events', subtitle: 'User centric feature coming soon** so people can manage their everyday vehicle checkups'),
            ],
          ),
        ),
      ),
    );
  }
}

class _ComponentItem extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;

  const _ComponentItem({required this.icon, required this.title, required this.subtitle});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF16161A),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white10),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(color: Colors.cyan.withValues(alpha: 0.1), borderRadius: BorderRadius.circular(8)),
            child: Icon(icon, color: Colors.cyan, size: 20),
          ),
          const SizedBox(width: 16),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
              Text(subtitle, style: const TextStyle(color: Colors.grey, fontSize: 12)),
            ],
          ),
        ],
      ),
    );
  }
}
