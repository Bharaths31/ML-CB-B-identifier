#!/usr/bin/env python3
"""
Single-command launcher for the Cattle & Buffalo Breed Classifier.

Starts both the Flask API backend and the React/Vite frontend server,
handles health-checks, displays URLs, and ensures clean shutdown.

Usage:
    python run.py                   # Run both servers & open browser
    python run.py --no-browser      # Run without auto-opening browser
    python run.py --backend-only    # Run only Flask backend (port 5000)
    python run.py --frontend-only   # Run only React frontend (port 5173)
"""

import os
import sys
import time
import signal
import socket
import shutil
import argparse
import webbrowser
import subprocess
from urllib.request import urlopen
from urllib.error import URLError

def _find_project_root() -> str:
    for start in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        cur = os.path.abspath(start)
        while cur:
            if os.path.isfile(os.path.join(cur, "src", "config.py")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    return os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = _find_project_root()
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "breed-classifier-react")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
try:
    os.chdir(PROJECT_ROOT)
except Exception:
    pass

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a network port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def get_python_executable() -> str:
    """Resolve virtual environment python if present."""
    if sys.platform == "win32":
        venv_python = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    if os.path.isfile(venv_python) and os.access(venv_python, os.X_OK):
        return venv_python
    return sys.executable


def get_npm_executable() -> str:
    """Resolve npm, including the Windows .cmd shim."""
    if sys.platform == "win32":
        return shutil.which("npm.cmd") or shutil.which("npm") or "npm.cmd"
    return shutil.which("npm") or "npm"

def wait_for_service(url: str, timeout_sec: float = 20.0, service_name: str = "Service") -> bool:
    """Poll an HTTP endpoint until it answers 200 OK or timeout expires."""
    start = time.time()
    print(f"Waiting for {service_name} at {url}...", end="", flush=True)
    while time.time() - start < timeout_sec:
        try:
            with urlopen(url, timeout=1.0) as res:
                if res.status in (200, 304):
                    print(" Ready!")
                    return True
        except (URLError, ConnectionError, OSError):
            pass
        print(".", end="", flush=True)
        time.sleep(0.5)
    print(" Timed out.")
    return False

def main():
    parser = argparse.ArgumentParser(
        description="Launch Breed Classifier Backend & Frontend Web Servers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backend-port", type=int, default=5000, help="Port for Flask API backend")
    parser.add_argument("--frontend-port", type=int, default=5173, help="Port for React Vite frontend")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")
    parser.add_argument("--backend-only", action="store_true", help="Launch only the Flask API server")
    parser.add_argument("--frontend-only", action="store_true", help="Launch only the Vite React frontend")
    parser.add_argument("--open-dev", action="store_true", help="Open developer page in browser alongside user page")
    args = parser.parse_args()

    python_bin = get_python_executable()
    procs = []

    def shutdown(signum=None, frame=None):
        print("\n\nShutting down all servers gracefully...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
        print("All servers stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("\n" + "=" * 68)
    print("  BREED CLASSIFIER — UNIFIED APPLICATION SERVER")
    print("=" * 68)

    # 1. Start Backend Server
    if not args.frontend_only:
        if is_port_in_use(args.backend_port):
            print(f"[!] Port {args.backend_port} is already in use. Reusing existing backend service.")
        else:
            print(f"[*] Starting Flask API backend on port {args.backend_port}...")
            env = os.environ.copy()
            env["PYTHONPATH"] = PROJECT_ROOT
            flask_cmd = [
                python_bin,
                os.path.join(PROJECT_ROOT, "server", "app.py"),
                "--port", str(args.backend_port),
            ]
            backend_proc = subprocess.Popen(
                flask_cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=None,
                stderr=None,
            )
            procs.append(backend_proc)

            backend_ready = wait_for_service(
                f"http://localhost:{args.backend_port}/api/health",
                timeout_sec=15.0,
                service_name="Backend API",
            )
            if not backend_ready:
                print("[!] Warning: Backend healthcheck timed out. Proceeding anyway.")

    # 2. Start Frontend Server
    if not args.backend_only:
        if not os.path.isdir(FRONTEND_DIR):
            print(f"[Error] Frontend directory not found at: {FRONTEND_DIR}")
            shutdown()

        if is_port_in_use(args.frontend_port):
            print(f"[!] Port {args.frontend_port} is already in use. Reusing existing frontend server.")
        else:
            print(f"[*] Starting React Vite frontend on port {args.frontend_port}...")
            npm_cmd = ["npm", "run", "dev", "--", "--port", str(args.frontend_port)]
            frontend_proc = subprocess.Popen(
                npm_cmd,
                cwd=FRONTEND_DIR,
                stdout=None,
                stderr=None,
            )
            procs.append(frontend_proc)

            frontend_ready = wait_for_service(
                f"http://localhost:{args.frontend_port}/",
                timeout_sec=15.0,
                service_name="Vite Frontend",
            )
            if not frontend_ready:
                print("[!] Warning: Frontend healthcheck timed out.")

    # 3. Banner & Browser Launch
    user_url = f"http://localhost:{args.frontend_port}/"
    dev_url = f"http://localhost:{args.frontend_port}/dev"
    api_url = f"http://localhost:{args.backend_port}/api/health"

    print("\n" + "-" * 68)
    print("  SERVICES RUNNING & READY:")
    print(f"  * User Portal:       {user_url}")
    print(f"  * Developer Portal:  {dev_url}")
    print(f"  * Backend REST API:  {api_url}")
    print("  * Shared Config DB:  outputs/app_config.json")
    print("-" * 68)
    print("  Press Ctrl+C to terminate all services\n")

    if not args.no_browser and not args.backend_only:
        time.sleep(0.5)
        webbrowser.open(user_url)
        if args.open_dev:
            time.sleep(0.3)
            webbrowser.open(dev_url)

    # 4. Monitor loop
    try:
        while True:
            for p in procs:
                if p.poll() is not None:
                    print(f"\n[!] A server process (PID {p.pid}) exited with code {p.returncode}.")
                    shutdown()
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()

if __name__ == "__main__":
    main()
