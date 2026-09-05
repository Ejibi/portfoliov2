import 'package:flutter/material.dart';

class AccountPage extends StatelessWidget {
  const AccountPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(48.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Account', style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold)),
          const Text('MANAGE DEVELOPER PROFILE & USAGE METRICS', style: TextStyle(color: Colors.grey, fontSize: 10, letterSpacing: 1.5)),
          const SizedBox(height: 48),
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: const Color(0xFF16161A),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.white10),
            ),
            child: Row(
              children: [
                const CircleAvatar(
                  radius: 32,
                  backgroundImage: NetworkImage('https://i.pravatar.cc/150?u=marcus'),
                ),
                const SizedBox(width: 24),
                const Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Marcus Vance PRO DEV', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                    Text('marcus.vance@neural-classify.io', style: TextStyle(color: Colors.grey)),
                    Text('Account created: November 12, 2023', style: TextStyle(color: Colors.grey, fontSize: 12)),
                  ],
                ),
                const Spacer(),
                ElevatedButton.icon(
                  onPressed: () {},
                  icon: const Icon(Icons.edit, size: 16, color: Colors.black),
                  label: const Text('Edit Profile', style: TextStyle(color: Colors.black)),
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.cyan),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),
          Row(
            children: [
              Expanded(
                child: Container(
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    color: const Color(0xFF16161A),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.white10),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Row(
                        children: [
                          Icon(Icons.lock_outline, color: Colors.cyan, size: 20),
                          SizedBox(width: 12),
                          Text('Security & Authorization', style: TextStyle(fontWeight: FontWeight.bold)),
                        ],
                      ),
                      const SizedBox(height: 24),
                      _AccountSettingRow(
                        label: 'Password Settings',
                        trailing: ElevatedButton(onPressed: () {}, child: const Text('Change Password')),
                      ),
                      _AccountSettingRow(
                        label: 'Two-Factor Auth (2FA)',
                        trailing: Switch(value: true, onChanged: (_) {}, activeThumbColor: Colors.cyan),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(width: 24),
              Expanded(
                child: Container(
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    color: const Color(0xFF16161A),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.red.withValues(alpha: 0.1)),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Row(
                        children: [
                          Icon(Icons.warning_amber_outlined, color: Colors.red, size: 20),
                          SizedBox(width: 12),
                          Text('Danger Zone', style: TextStyle(color: Colors.red, fontWeight: FontWeight.bold)),
                        ],
                      ),
                      const Divider(color: Colors.red, height: 32),
                      const Text('Delete Developer Account', style: TextStyle(fontWeight: FontWeight.bold)),
                      const Text('Permanently purge Marcus Vance profile, datasets, classification history, and metadata arrays.', style: TextStyle(color: Colors.grey, fontSize: 12)),
                      const SizedBox(height: 16),
                      ElevatedButton(
                        onPressed: () {},
                        style: ElevatedButton.styleFrom(backgroundColor: Colors.transparent, side: const BorderSide(color: Colors.red)),
                        child: const Text('Delete Account', style: TextStyle(color: Colors.red)),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _AccountSettingRow extends StatelessWidget {
  final String label;
  final Widget trailing;

  const _AccountSettingRow({required this.label, required this.trailing});

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
