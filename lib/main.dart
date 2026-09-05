import 'package:flutter/material.dart';
import 'navigation_shell.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  // Sign in anonymously for zero-friction auth metrics
  await FirebaseAuth.instance.signInAnonymously();
  runApp(const AiPortfolioApp());
}

class AiPortfolioApp extends StatelessWidget {
  const AiPortfolioApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Portfolio_v2',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF0D0D0F),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF00E5FF), // Cyan accent
          secondary: Color(0xFF00E5FF),
          surface: Color(0xFF16161A),
          onSurface: Colors.white,
        ),
        textTheme: const TextTheme(
          displayMedium: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
          bodyLarge: TextStyle(color: Color(0xFFA0A0A5)),
        ),
        useMaterial3: true,
      ),
      home: const RootNavigation(),
    );
  }
}
