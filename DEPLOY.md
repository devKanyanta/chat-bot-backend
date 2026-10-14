# Deploying the ZedChatBot backend on a VPS

Step-by-step host setup for the FastAPI backend behind **PM2 + nginx + PostgreSQL**.
Written against commit `7243a7c` of `chat-bot-backend`. Start at section 2 for the
database; sections 0–1 are the one-time machine prep.

Assumed: a fresh Debian/Ubuntu VPS, a sudo-capable user, PM2 already installed, and a
domain (or subdomain) whose DNS A record points at the server's IP.

---

## 0. What the VPS needs

- **1 vCPU / 1 GB RAM** is plenty. This app is I/O bound — it waits on Google's API and
  on PostgreSQL; it does no heavy local computation.
- **Open ports:** 22 (SSH), 80 and 443 (nginx/certbot). Leave **8000** and **5432** closed
  to the internet; both stay bound to `127.0.0.1`.
- **A hostname** such as `api.yourdomain.com`. Required for TLS, which the mobile app
  needs (see section 9).

## 1. Base packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv python3-pip git nginx postgresql
sudo apt install -y certbot python3-certbot-nginx

python3 -V      # the app is developed on 3.14; every pinned dependency declares >=3.10
node -v && pm2 -v
```

If `python3 -V` reports anything older than 3.10, install a newer Python before
continuing — the pinned FastAPI/Starlette releases require it.

**Fix name resolution while you are here.** Every chat request opens an outbound
connection to `generativelanguage.googleapis.com`. On your laptop that host once failed
to resolve mid-chat (`Temporary failure in name resolution` → HTTP 502 to the phone),
because the only resolver was the router. Give the server public resolvers:

```bash
sudo mkdir -p /etc/systemd/resolved.conf.d
sudo tee /etc/systemd/resolved.conf.d/dns.conf >/dev/null <<'CONF'
[Resolve]
DNS=1.1.1.1 8.8.8.8
FallbackDNS=9.9.9.9
CONF
sudo systemctl restart systemd-resolved
resolvectl status | head -20        # confirm the new servers are listed
```

## 2. Database (start here)

Install once, then create a role and database. Generate the password first so it does not
end up in your shell history or in a chat log:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(24))"
```

```bash
sudo -u postgres psql <<'SQL'
CREATE ROLE zedchatbot WITH LOGIN PASSWORD 'PASTE_GENERATED_PASSWORD_HERE';
CREATE DATABASE zedchatbot OWNER zedchatbot ENCODING 'UTF8';
SQL
```

Making `zedchatbot` the database owner matters: from PostgreSQL 15 on, the `public`
schema is not writable by other roles, and the app creates its tables itself.

Verify password auth over TCP (Debian/Ubuntu's default `pg_hba.conf` already allows
`scram-sha-256` on `127.0.0.1`):

```bash
PGPASSWORD='PASTE_GENERATED_PASSWORD_HERE' \
  psql -h 127.0.0.1 -U zedchatbot -d zedchatbot -c 'SELECT current_database(), current_user;'
```

**There is no migration step.** `app/db.py:init_db()` calls `Base.metadata.create_all()`
during FastAPI startup, so the tables appear the first time the app boots (section 5).
The project deliberately has no Alembic — noted as a limitation in the backend spec, and
worth knowing: `create_all` only creates *missing tables*. If you later change a column
in `app/models.py`, existing tables are left untouched and you must write the `ALTER
TABLE` by hand.

**Optional — bring your local data across.** The dump carries registered users (bcrypt
password hashes still work), conversations and messages:

```bash
# on the laptop
pg_dump -h 127.0.0.1 -U zedchatbot -d zedchatbot -Fc -f zedchatbot.dump
scp zedchatbot.dump you@your-vps:/tmp/

# on the VPS
pg_restore -h 127.0.0.1 -U zedchatbot -d zedchatbot --no-owner --clean --if-exists /tmp/zedchatbot.dump
rm /tmp/zedchatbot.dump
```

Existing access tokens stop working, because production uses a different `JWT_SECRET`
(section 4) — everyone signs in again. Passwords and chat history survive.

## 3. Code and virtual environment

```bash
sudo mkdir -p /srv/zedchatbot && sudo chown "$USER":"$USER" /srv/zedchatbot
git clone https://github.com/devKanyanta/chat-bot-backend.git /srv/zedchatbot/chat-bot-backend
cd /srv/zedchatbot/chat-bot-backend

python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
git log -1 --oneline      # expect 7243a7c (retry/fallback + AFC fixes)
```

If the repository is private, use a **read-only deploy key**, not a personal access
token:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/gh_deploy -N ""
cat ~/.ssh/gh_deploy.pub      # add at: GitHub repo -> Settings -> Deploy keys (read-only)
git clone git@github.com:devKanyanta/chat-bot-backend.git /srv/zedchatbot/chat-bot-backend
```

## 4. Production `.env`

```bash
cd /srv/zedchatbot/chat-bot-backend
cp .env.example .env
```

The values that must differ from local development:

| Key | Production value | Why |
|---|---|---|
| `GEMINI_API_KEY` | a real key | Chat is dead without it; `/health` reports `gemini_configured`. |
| `GEMINI_MODEL` | e.g. `gemini-3.8-flash` | Keep the model that has been answering. |
| `GEMINI_FALLBACK_MODELS` | `gemini-3.7-flash,gemini-3.6-flash` | Tried in order on 503/404/429 so one saturated model does not take chat down. |
| `DATABASE_URL` | `postgresql+psycopg://zedchatbot:PASSWORD@127.0.0.1:5432/zedchatbot` | Postgres, over loopback. |
| `JWT_SECRET` | **new** value | The built-in default in `app/config.py` is public; leaving it in place makes every login token forgeable. |
| `CORS_ORIGINS` | `https://your-frontend-domain` | Only the browser frontend needs CORS. |
| `DB_ECHO` | `false` | Otherwise every SQL statement is logged. |

Generate the secret with `python3 -c "import secrets; print(secrets.token_urlsafe(48))"`.

Two traps specific to this project:

- Your laptop's `.env` declares `DATABASE_URL` **twice** (a `sqlite:///` line first, then
  the Postgres one). Pydantic uses the **last** occurrence, so things work — but write
  exactly one line in production. Confirmed locally: the effective backend is
  `postgresql+psycopg`.
- `.env` is resolved **relative to the process working directory** (`env_file=".env"` in
  `app/config.py`). That is why the PM2 config pins `cwd`; a process started from `/`
  would silently fall back to the SQLite default.

```bash
chmod 600 .env      # already gitignored — never commit it
```

## 5. Smoke test before PM2

Run it in the foreground, exactly as the service manager will:

```bash
cd /srv/zedchatbot/chat-bot-backend
./venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another shell:

```bash
curl -s http://127.0.0.1:8000/health
```

Expect `{"status":"ok","database_connected":true,...,"database_backend":"postgresql+psycopg"}`.
Then press Ctrl-C: startup should log `Application shutdown complete`. This is the same
command line the PM2 config uses, so a failure here is a config/env problem, not a PM2 one.

## 6. PM2

```bash
cd /srv/zedchatbot/chat-bot-backend
pm2 start ecosystem.config.cjs
pm2 status
pm2 logs zedchatbot-backend --lines 50
pm2 save
pm2 startup          # run the sudo command it prints, once
```

Three things the config is doing deliberately:

- **One instance, fork mode.** The per-IP rate limiter (`app/ratelimit.py`) and the
  retention sweeper (`app/retention.py`) hold state in process memory. A second worker
  would double the real rate limit per client and run duplicate sweeps. The backend spec
  flags this: multi-worker needs a shared store first.
- **`--proxy-headers --forwarded-allow-ips=127.0.0.1`.** Without it, nginx connections
  all look like `127.0.0.1` and every internet user shares one 30 req/min bucket. Trusting
  only the local proxy means a client cannot spoof its own address.
- **`kill_timeout: 15000`.** A restart still ends in-flight SSE replies — deploy when
  nobody is mid-answer, or accept that a generating reply is lost.

## 7. nginx and TLS

```bash
cd /srv/zedchatbot/chat-bot-backend
sudo cp deploy/nginx-zedchatbot.conf /etc/nginx/sites-available/zedchatbot
sudo sed -i 's/api\.example\.com/api.yourdomain.com/g' /etc/nginx/sites-available/zedchatbot
sudo ln -sf /etc/nginx/sites-available/zedchatbot /etc/nginx/sites-enabled/zedchatbot
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

curl -s http://api.yourdomain.com/health      # through the proxy, before TLS
sudo certbot --nginx -d api.yourdomain.com
```

certbot edits the file in place, adding the TLS block and an HTTP→HTTPS redirect. Confirm
it kept the streaming settings:

```bash
grep -n 'proxy_read_timeout\|proxy_buffering' /etc/nginx/sites-available/zedchatbot
# proxy_buffering off;  proxy_read_timeout 900s;  proxy_send_timeout 900s;
```

Those lines exist because nginx's defaults are hostile to this app: `proxy_read_timeout`
is **60 s** and body buffering is on, while a chat reply here streams for 47–294 s under
current Gemini load. Left at defaults, nginx cuts the connection at 60 s and the user sees
a truncated or failed answer even though the backend was working. The backend also sends
`X-Accel-Buffering: no` on that route; the config makes it explicit regardless.

Verify end to end, from outside, watching the stream arrive:

```bash
curl -s https://api.yourdomain.com/health
curl -N -s -X POST https://api.yourdomain.com/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"Reply with the single word OK"}'
```

`-N` disables curl's own buffering so you can see `data:` lines appear as they are
generated. To check the mobile path, replace the URL with the phone's expectation.

## 8. Firewall

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable && sudo ufw status

ss -ltnp | grep -E ':8000|:5432'      # both must show 127.0.0.1, never 0.0.0.0
```

Optional but cheap on a public host: `sudo apt install -y fail2ban`.

## 9. Point the clients at it

**Flutter app**

```bash
flutter run --dart-define=API_BASE_URL=https://api.yourdomain.com
flutter build apk --dart-define=API_BASE_URL=https://api.yourdomain.com
```

This is the whole reason the VPS needs TLS: the cleartext-HTTP exception in the app lives
only in `android/app/src/debug/`, so a **release APK requires HTTPS**. Once this is live,
also drop the hardcoded LAN address from `defaultValue` in `lib/core/app_config.dart`
(`http://192.168.0.189:8000` is DHCP-assigned and breaks on the next lease change).

**React frontend:** point its API base (see `src/config.js`) at the same URL and add that
origin to `CORS_ORIGINS`. The mobile app sends no `Origin` header, so CORS never applies
to it.

## 10. Backups

`zedchatbot` is a **database role, not an OS account** — there is no `zedchatbot` Unix user to run things as. The cron job below runs as root, so root needs the credential file. Open it in an editor rather than echoing the password (which would land in your shell history):

```bash
sudo nano /root/.pgpass     # paste this single line, with the real password:
# 127.0.0.1:5432:zedchatbot:zedchatbot:YOUR_PASSWORD
sudo chmod 600 /root/.pgpass
```

```bash
sudo crontab -e     # daily 02:30 dump, keep 14 days
30 2 * * * pg_dump -h 127.0.0.1 -U zedchatbot -Fc zedchatbot | gzip > /var/backups/zedchatbot-$(date +\%F).dump.gz && find /var/backups -name 'zedchatbot-*.dump.gz' -mtime +14 -delete
```

Restore-test it once into a throwaway database — do not use `pg_restore --create` here, the archive already carries the `zedchatbot` name and it would collide:

```bash
createdb -h 127.0.0.1 -U zedchatbot zedchatbot_restore
gunzip -c /var/backups/zedchatbot-2026-10-14.dump.gz | \
  pg_restore -h 127.0.0.1 -U zedchatbot -d zedchatbot_restore --no-owner
psql -h 127.0.0.1 -U zedchatbot -d zedchatbot_restore -c '\dt'
dropdb -h 127.0.0.1 -U zedchatbot zedchatbot_restore
```

An untested backup is not a backup.

## 11. Day-2 operations

| Task | Command |
|---|---|
| Liveness | `curl -s https://api.yourdomain.com/health` |
| App logs | `pm2 logs zedchatbot-backend --lines 100` |
| Restart | `pm2 restart zedchatbot-backend` |
| Deploy an update | `cd /srv/zedchatbot/chat-bot-backend && git pull && ./venv/bin/pip install -r requirements.txt && pm2 restart zedchatbot-backend` |
| Rotate the Gemini key | edit `.env`, then `pm2 restart zedchatbot-backend` (the app reads `.env` at startup) |
| nginx errors | `sudo tail -f /var/log/nginx/error.log` |
| DB size | `sudo -u postgres psql -c '\l+'` |
| Log growth | `pm2 install pm2-logrotate` |

## 12. Known limits to expect

- **No migrations.** `create_all` only adds missing tables; schema changes are manual SQL.
- **One worker by design** (see section 6). Horizontal scaling needs a shared rate-limit
  store first.
- **Rate limits reset on restart**, since the counter lives in memory.
- **Latency is Google's.** The backend sends no response headers until the first token
  arrives, so a phone can hit the app's own `STREAM_TIMEOUT_SECONDS` (180 s) on a request
  the server would eventually complete. nginx now allows 900 s, which means the client
  limit is the binding one — raise it with
  `--dart-define=STREAM_TIMEOUT_SECONDS=300` if slow replies keep failing on the phone.
- **`/health` is the only probe** — no metrics, no alerting.
- **Sweeper runs in-process**; while the app is down, expired anonymous conversations are
  simply cleaned up at the next cycle instead of on schedule.
