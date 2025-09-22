#!/usr/bin/env python3
"""
Employee Analytics Dashboard - Application Runner
==================================================

This file serves as the entry point for the Employee Analytics Dashboard.
It sets up the Flask application with proper configuration and runs it.

Usage:
    python run.py                    # Development mode
    python run.py --prod            # Production mode
    python run.py --host 0.0.0.0   # Custom host
    python run.py --port 8080      # Custom port

Environment Variables:
    FLASK_ENV: development/production
    FLASK_HOST: Host to bind to (default: 127.0.0.1)
    FLASK_PORT: Port to bind to (default: 5000)
    FLASK_DEBUG: Enable debug mode (default: True in development)
"""

import os
import sys
import argparse
from app import create_app


def main():
    """Main application runner"""
    parser = argparse.ArgumentParser(description='Employee Analytics Dashboard')
    parser.add_argument('--host', default=None, help='Host to bind to')
    parser.add_argument('--port', type=int, default=None, help='Port to bind to')
    parser.add_argument('--prod', action='store_true', help='Run in production mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    # Determine configuration
    if args.prod:
        os.environ['FLASK_ENV'] = 'production'
        config_name = 'production'
    else:
        os.environ['FLASK_ENV'] = 'development'
        config_name = 'development'

    # Create Flask application
    app = create_app()

    # Configure host and port
    host = args.host or os.environ.get('FLASK_HOST', '127.0.0.1')
    port = args.port or int(os.environ.get('FLASK_PORT', 5000))
    debug = args.debug or (config_name == 'development')

    # Print startup information
    print("=" * 60)
    print("🚀 Employee Analytics Dashboard")
    print("=" * 60)
    print(f"📊 Environment: {config_name}")
    print(f"🌐 URL: http://{host}:{port}")
    print(f"🔧 Debug Mode: {'Enabled' if debug else 'Disabled'}")
    print("=" * 60)
    print("Features Available:")
    print("✅ Interactive Charts (Pie, Bar, Scatter, Heatmap, Radar)")
    print("✅ Dynamic Filtering & Search")
    print("✅ Individual Employee Analysis")
    print("✅ Performance Predictions & Risk Assessment")
    print("✅ Report Generation & Export")
    print("✅ Sample Data Generator")
    print("=" * 60)
    print("📁 Supported File Formats: Excel (.xlsx), CSV (.csv)")
    print("🎯 Ready for your company's employee analytics!")
    print("=" * 60)

    try:
        # Run the application
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down Employee Analytics Dashboard...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()