// PM2 process definition for the ZedChatBot FastAPI backend.
//
//   pm2 start ecosystem.config.cjs
//   pm2 logs zedchatbot-backend
//   pm2 save            # remember this process list across reboots
//   pm2 startup         # run the sudo command it prints, once
//
// Why one instance and not PM2 cluster mode: this app keeps two pieces of
// state in process memory (see zedchatbot-backend-spec.md, risks section) —
// the per-IP rate limiter in app/ratelimit.py and the retention sweeper task
// in app/retention.py. Extra workers would multiply the effective rate limit
// per client and run duplicate sweeps. Scale up before scaling out.
const fs = require('node:fs');
const path = require('node:path');

const APP_DIR = __dirname; // .../chat-bot-backend
const LOG_DIR = path.join(APP_DIR, 'logs');

// PM2 creates the log files but not their directory.
fs.mkdirSync(LOG_DIR, { recursive: true });

const HOST = process.env.BIND_HOST || '127.0.0.1';
const PORT = process.env.BIND_PORT || '8000';

module.exports = {
  apps: [
    {
      name: 'zedchatbot-backend',

      // Run the venv's uvicorn directly — no shell, no node interpreter.
      script: path.join(APP_DIR, 'venv', 'bin', 'uvicorn'),
      interpreter: 'none',
      args: [
        'app.main:app',
        '--host', HOST,          // private bind: nginx is the public entry point
        '--port', PORT,
        '--workers', '1',        // see the note at the top of this file
        // Trust X-Forwarded-For only from the local nginx, so the per-IP rate
        // limiter sees real client addresses instead of 127.0.0.1 for everyone.
        '--proxy-headers',
        '--forwarded-allow-ips', '127.0.0.1',
        '--timeout-keep-alive', '15',
      ].join(' '),

      // .env is resolved relative to the process working directory.
      cwd: APP_DIR,

      instances: 1,
      exec_mode: 'fork',
      watch: false,
      autorestart: true,
      min_uptime: '20s',        // a process dying faster than this counts as a crash
      max_restarts: 15,         // stop flapping; recover with `pm2 restart`
      restart_delay: 3000,
      // Give in-flight SSE streams a moment to close on shutdown.
      kill_timeout: 15000,

      merge_logs: true,
      time: true,
      out_file: path.join(LOG_DIR, 'out.log'),
      error_file: path.join(LOG_DIR, 'error.log'),

      env: {
        PYTHONUNBUFFERED: '1',      // let uvicorn logs reach PM2 immediately
        PYTHONDONTWRITEBYTECODE: '1',
      },
    },
  ],
};
