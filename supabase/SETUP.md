# Supabase project setup (one-time, manual)

These steps create the Supabase project, run the migrations, and create
the admin user. They cannot be automated — they require dashboard access.

## 1. Create the project

1. Go to https://supabase.com/dashboard, sign in, click "New project".
2. Name: `badminton-tracker` (or whatever).
3. Region: closest to you and to the Modal region you use.
4. Pick a strong DB password and store it in your password manager.
5. Wait for the project to provision (~1 minute).

## 2. Capture credentials

From Project Settings → API:
- `SUPABASE_URL` — Project URL.
- `SUPABASE_ANON_KEY` — anon/public key.
- `SUPABASE_SERVICE_ROLE_KEY` — service_role key. **Treat as a secret.
  Never commit it; never put it in client bundles.**

Store them somewhere safe (1Password, etc.) — you'll paste them into
.env files and Modal Secrets later.

## 3. Run migrations

Two options. Pick one.

### Option A — `supabase` CLI (preferred)

From the repo root:

    supabase link --project-ref <your-project-ref>
    supabase db push

This applies every file in `supabase/migrations/` in order.

### Option B — SQL Editor (manual)

For each file in `supabase/migrations/` in order:
1. Open the SQL Editor in the dashboard.
2. Paste the file contents.
3. Run.

## 4. Configure Auth

In Authentication → Providers:

1. **Email**:
   - Enable.
   - Confirm email: ON.
   - **Disable signups**: ON (so only you can create users from the dashboard).
2. **Google** (optional but recommended):
   - Get a Google Cloud OAuth client ID/secret. (Google Cloud Console →
     APIs & Services → Credentials → Create OAuth 2.0 client ID; web type;
     authorized redirect URI: `https://<project-ref>.supabase.co/auth/v1/callback`).
   - Paste client ID and secret into Supabase Auth → Providers → Google.
   - Enable.

## 5. Create your admin user

In Authentication → Users:

1. Click "Add user" → "Send invite".
2. Enter your email.
3. Open the invite email and click the link to set your password.
4. You're now logged in via the Supabase studio with that user.

## 6. Hand off credentials to the implementation

Paste these into the project's `.env.local` (frontend) and into Modal
Secrets (backend) per the plan tasks that come next.
