#!/usr/bin/env python3
"""
Startup Cold-Email Automation
==============================
Finds recently-funded iOS-relevant startups and sends personalized cold emails
to founders. Runs daily via GitHub Actions at 09:00 UTC.

Sources  : YC Company Directory (public JSON API) + TechCrunch RSS
Filter   : keywords — mobile, consumer, app, ios, iphone
Enrich   : Apollo.io API (primary) → email-pattern guessing + SMTP verify (fallback)
Personalize: Google Gemini Flash (2-sentence pitch per company)
Send     : Gmail SMTP (App Password) with CV attached as PDF
Track    : leads.csv committed back to the repo
"""

import csv
import logging
import os
import re
import smtplib
import time
from datetime import datetime, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import urlparse

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import google.generativeai as genai

# ── Bootstrap ──────────────────────────────────────────────────────────────────

load_dotenv()  # no-op in GitHub Actions; useful for local dev with a .env file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

APOLLO_API_KEY: str = os.environ.get("APOLLO_API_KEY", "")
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GMAIL_USER: str = os.environ.get("GMAIL_USER", "")
GMAIL_PASS: str = os.environ.get("GMAIL_PASS", "")

LEADS_CSV = "leads.csv"
CV_PATH = "Sharang_Verma_iOS_Engineer.pdf"   # Commit this PDF to the repo root
MAX_NEW_LEADS_PER_RUN = 10  # Safety cap — Apollo free tier ≈ 50 enrichments/month

FILTER_KEYWORDS = [
    "mobile", "consumer", "app", "ios", "iphone",
    "social", "marketplace", "fintech", "wallet", "checkout",
]

# ── Sender persona (from resume) ───────────────────────────────────────────────

SENDER_NAME = "Sharang Verma"
SENDER_EMAIL = "sharangverma2002@gmail.com"
SENDER_LINKEDIN = "linkedin.com/in/sharang-verma"
SENDER_GITHUB = "github.com/Sharang-1"

# Context fed to Gemini when generating pitches
SENDER_CONTEXT = (
    "iOS Engineer, 2+ years experience. "
    "Most recently at Pincode (PhonePe) — consumer app with millions of users. "
    "Previously at AiDash (B2B enterprise, built geo-mapping app from scratch) and DZOR. "
    "Open-sourced SwiftModelGraph (Swift Macros + IndexStoreDB). "
    "Skills: Swift, UIKit, SwiftUI, MapKit, KMP, StoreKit, A/B testing, CI/CD."
)

STATIC_PITCH_FALLBACK = (
    "A polished, performant iOS experience is often the deciding factor between "
    "retention and churn — especially for consumer-facing products. "
    "I'd love to bring that focus to your team as a remote iOS engineer."
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data Sourcing
# ══════════════════════════════════════════════════════════════════════════════

def fetch_yc_companies(max_pages: int = 5) -> list[dict]:
    """
    Fetch companies from the public YC Company Directory API.
    Returns a list of normalised company dicts.
    """
    companies: list[dict] = []
    base_url = "https://api.ycombinator.com/v0.1/companies"

    for page in range(1, max_pages + 1):
        try:
            resp = requests.get(
                base_url,
                params={"page": page, "per_page": 100},
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("companies", [])
            if not items:
                break

            for item in items:
                website = item.get("website") or ""
                companies.append({
                    "name": item.get("name", "").strip(),
                    "domain": extract_domain(website),
                    "description": (
                        f"{item.get('one_liner', '')} "
                        f"{item.get('long_description', '')}"
                    ).strip(),
                    "source": "yc",
                })

            log.info(f"YC page {page}: +{len(items)} companies")

        except requests.HTTPError as exc:
            log.warning(f"YC API HTTP error page {page}: {exc}")
            break
        except Exception as exc:
            log.warning(f"YC API error page {page}: {exc}")
            break

    log.info(f"YC source total: {len(companies)}")
    return companies


def fetch_rss_companies() -> list[dict]:
    """
    Parse TechCrunch Startups / Funding RSS feeds.
    Extracts a company domain + description from each article.
    """
    rss_feeds = [
        "https://techcrunch.com/category/startups/feed/",
        "https://techcrunch.com/tag/funding/feed/",
    ]

    # Domains that appear in RSS but are NOT startup company sites
    NOISE_DOMAINS = {
        "techcrunch.com", "crunchbase.com", "twitter.com", "x.com",
        "linkedin.com", "facebook.com", "youtube.com", "instagram.com",
        "apple.com", "google.com", "microsoft.com", "amazon.com",
        "t.co", "bit.ly", "ow.ly",
    }

    companies: list[dict] = []

    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            if feed.bozo and not feed.entries:
                log.warning(f"RSS feed unavailable or malformed: {feed_url}")
                continue

            for entry in feed.entries:
                title: str = entry.get("title", "")
                raw_summary: str = (
                    entry.get("summary") or entry.get("description") or ""
                )
                # Strip HTML tags from summary
                summary = BeautifulSoup(raw_summary, "html.parser").get_text(
                    separator=" "
                ).strip()

                domain = _extract_best_domain_from_rss(title, raw_summary, NOISE_DOMAINS)
                if not domain:
                    continue

                # Heuristic: derive company name from article title
                name = _extract_company_name_from_title(title)

                companies.append({
                    "name": name,
                    "domain": domain,
                    "description": summary[:1000],
                    "source": "rss",
                })

        except Exception as exc:
            log.warning(f"RSS error ({feed_url}): {exc}")

    log.info(f"RSS source total: {len(companies)}")
    return companies


def _extract_best_domain_from_rss(
    title: str, html_content: str, noise: set[str]
) -> str:
    """Pull the first non-noise domain found in the HTML content or title."""
    url_re = re.compile(
        r'https?://(?:www\.)?([a-zA-Z0-9\-]+\.[a-zA-Z]{2,})(?:/[^\s"\'<>]*)?'
    )
    for match in url_re.finditer(html_content):
        domain = match.group(1).lower()
        if domain not in noise:
            return domain
    return ""


def _extract_company_name_from_title(title: str) -> str:
    """
    Best-effort company name extraction from a TC headline like
    'Acme raises $10M Series A to build X'.
    """
    for splitter in [" raises ", " lands ", " secures ", " closes ", " gets ", " nabs "]:
        if splitter in title.lower():
            return title[: title.lower().index(splitter)].strip()
    # Fallback: first segment before a colon or dash
    for sep in [":", " — ", " - "]:
        if sep in title:
            return title.split(sep)[0].strip()
    return title.strip()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Filtering & Deduplication
# ══════════════════════════════════════════════════════════════════════════════

def filter_companies(companies: list[dict]) -> list[dict]:
    """Keep only companies whose name or description mentions target keywords."""
    filtered = [
        c for c in companies
        if any(
            kw in (f"{c.get('name','')} {c.get('description','')}").lower()
            for kw in FILTER_KEYWORDS
        )
    ]
    log.info(f"Filter: {len(filtered)}/{len(companies)} match keywords")
    return filtered


def extract_domain(url: str) -> str:
    """Normalise a URL to a bare domain (no www, no path, no port)."""
    if not url:
        return ""
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url
    try:
        netloc = urlparse(url).netloc.lower().lstrip("www.")
        return netloc.split(":")[0]
    except Exception:
        return ""


def load_contacted_leads(filepath: str) -> set[str]:
    """Return the set of domains already in leads.csv."""
    contacted: set[str] = set()
    if not os.path.exists(filepath):
        return contacted
    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if d := row.get("domain", "").strip():
                contacted.add(d)
    log.info(f"Loaded {len(contacted)} previously-contacted domains")
    return contacted


def init_leads_csv(filepath: str) -> None:
    """Create leads.csv with headers if it doesn't exist yet."""
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["timestamp", "domain", "company_name", "contact_email", "status"]
            )
        log.info(f"Created {filepath}")


def save_lead(
    filepath: str,
    domain: str,
    company_name: str,
    contact_email: str,
    status: str,
) -> None:
    """Append one row to leads.csv."""
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            domain,
            company_name,
            contact_email,
            status,
        ])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Enrichment: Apollo.io
# ══════════════════════════════════════════════════════════════════════════════

class ApolloRateLimitError(Exception):
    """Raised when Apollo returns HTTP 429 so tenacity can retry."""


@retry(
    retry=retry_if_exception_type(ApolloRateLimitError),
    wait=wait_exponential(multiplier=2, min=10, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _apollo_people_search_raw(domain: str) -> dict:
    """Low-level Apollo API call — retried by tenacity on 429."""
    resp = requests.post(
        "https://api.apollo.io/api/v1/people/search",
        json={
            "api_key": APOLLO_API_KEY,
            "q_organization_domains": domain,
            "person_titles": ["Founder", "Co-Founder", "CEO", "CTO", "Co-founder"],
            "per_page": 5,
        },
        timeout=20,
    )
    if resp.status_code == 429:
        log.warning("Apollo 429 — will retry with backoff")
        raise ApolloRateLimitError()
    resp.raise_for_status()
    return resp.json()


def find_contact_apollo(domain: str) -> dict | None:
    """
    Search Apollo for a Founder/CTO at the given domain.
    Returns a dict with keys: email (may be None), first_name, last_name, title.
    Returns None if no person found or on unrecoverable error.
    """
    if not APOLLO_API_KEY:
        log.warning("APOLLO_API_KEY not set — skipping Apollo lookup")
        return None

    try:
        data = _apollo_people_search_raw(domain)
    except ApolloRateLimitError:
        log.error(f"Apollo: rate limit exhausted for {domain}")
        return None
    except RetryError:
        log.error(f"Apollo: max retries exceeded for {domain}")
        return None
    except Exception as exc:
        log.warning(f"Apollo error for {domain}: {exc}")
        return None

    people = data.get("people", [])
    if not people:
        log.info(f"Apollo: no people found for {domain}")
        return None

    person = people[0]
    return {
        "email": person.get("email"),          # may be None on free tier
        "first_name": person.get("first_name", ""),
        "last_name": person.get("last_name", ""),
        "title": person.get("title", ""),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Enrichment: Email Guessing Fallback
# ══════════════════════════════════════════════════════════════════════════════

def scrape_website_email_pattern(domain: str) -> str | None:
    """
    Scrape the company site for any visible email addresses to infer their
    internal naming convention (e.g. 'first.last', 'f.last', 'first').
    Returns a pattern string or None.
    """
    GENERIC_LOCALS = {
        "info", "hello", "hi", "support", "help", "contact", "admin",
        "team", "sales", "press", "media", "jobs", "careers",
        "no-reply", "noreply", "billing", "legal", "feedback",
    }

    found_emails: list[str] = []

    for path in ["", "/contact", "/about", "/team"]:
        url = f"https://{domain}{path}"
        try:
            resp = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
                allow_redirects=True,
            )
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # mailto: links are the most reliable source
            for tag in soup.find_all("a", href=re.compile(r"^mailto:", re.I)):
                address = tag["href"][7:].split("?")[0].strip().lower()
                if "@" in address:
                    found_emails.append(address)

            # Plain-text email addresses anywhere on the page
            page_text = soup.get_text()
            for match in re.findall(
                r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
                page_text,
            ):
                found_emails.append(match.lower())

        except Exception:
            continue  # network errors on individual paths are normal

    # Keep only emails from this domain
    own = [e for e in found_emails if e.endswith(f"@{domain}")]
    if not own:
        return None

    # Find the first non-generic email to infer the pattern
    for email in own:
        local = email.split("@")[0]
        if local not in GENERIC_LOCALS:
            if re.fullmatch(r'[a-z]+\.[a-z]+', local):
                return "first.last"
            if re.fullmatch(r'[a-z]\.[a-z]+', local):
                return "f.last"
            if re.fullmatch(r'[a-z]{6,}', local):
                return "firstlast"
            if re.fullmatch(r'[a-z]{2,5}', local):
                return "first"

    return None


def guess_founder_email(
    first_name: str,
    last_name: str,
    domain: str,
    pattern: str | None,
) -> list[str]:
    """
    Generate ordered candidate email addresses for a founder.
    If a naming pattern was detected from the website, that format comes first.
    Returns a list of up to 4 unique candidates.
    """
    fn = first_name.lower().strip()
    ln = last_name.lower().strip()

    if not fn or not domain:
        return []

    fi = fn[0]  # first initial (e.g. "s" for Sharang)

    # Map: pattern-name → candidate address
    all_candidates: dict[str, str] = {
        "first.last":  f"{fn}.{ln}@{domain}" if ln else "",
        "f.last":      f"{fi}.{ln}@{domain}" if ln else "",
        "first":       f"{fn}@{domain}",
        "firstlast":   f"{fn}{ln}@{domain}" if ln else "",
    }

    ordered: list[str] = []

    # Detected pattern goes first
    if pattern and all_candidates.get(pattern):
        ordered.append(all_candidates[pattern])

    # Then add remaining non-empty candidates
    for key in ["first.last", "f.last", "first", "firstlast"]:
        candidate = all_candidates.get(key, "")
        if candidate and candidate not in ordered:
            ordered.append(candidate)

    return ordered


def verify_email_smtp(email: str) -> bool:
    """
    Best-effort SMTP verification via MX record lookup + RCPT TO probe.
    No email is actually sent. Returns False on any error (including servers
    that block this probing technique — very common).
    """
    try:
        import dns.resolver  # imported here to keep startup fast

        domain = email.split("@")[1]
        mx_records = dns.resolver.resolve(domain, "MX")
        mx_host = str(
            sorted(mx_records, key=lambda r: r.preference)[0].exchange
        ).rstrip(".")

        with smtplib.SMTP(mx_host, 25, timeout=10) as smtp:
            smtp.ehlo("verify.local")
            smtp.mail("probe@verify.local")
            code, _ = smtp.rcpt(email)
            return code == 250

    except Exception:
        return False


def resolve_contact_email(
    domain: str,
    apollo_result: dict | None,
) -> tuple[str | None, str, str]:
    """
    Full enrichment pipeline:
      1. Use Apollo-provided email if available.
      2. If Apollo found the person but withheld the email:
         a. Scrape site for naming pattern.
         b. Generate candidates.
         c. SMTP-verify each; use first verified hit.
         d. Fall back to best-guess candidate if SMTP verify fails for all.
      3. Return (email_or_None, first_name, last_name).
    """
    if not apollo_result:
        return None, "", ""

    first_name = apollo_result.get("first_name", "")
    last_name = apollo_result.get("last_name", "")

    # Best case: Apollo gave us the email
    if apollo_result.get("email"):
        return apollo_result["email"], first_name, last_name

    # Apollo found the person but withheld the email — try guessing
    if not first_name:
        log.info(f"Apollo returned no name for {domain} — cannot guess email")
        return None, first_name, last_name

    log.info(
        f"Apollo found {first_name} {last_name} at {domain} but no email "
        f"— attempting email pattern guess"
    )

    pattern = scrape_website_email_pattern(domain)
    if pattern:
        log.info(f"Detected email pattern for {domain}: '{pattern}'")

    candidates = guess_founder_email(first_name, last_name, domain, pattern)

    for candidate in candidates:
        if verify_email_smtp(candidate):
            log.info(f"SMTP verified ✓  {candidate}")
            return candidate, first_name, last_name

    # SMTP verify failed (most servers block it) — use top-confidence guess
    if candidates:
        log.info(f"SMTP verify inconclusive — using best-guess: {candidates[0]}")
        return candidates[0], first_name, last_name

    return None, first_name, last_name


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Personalization: Google Gemini
# ══════════════════════════════════════════════════════════════════════════════

def _gemini_model() -> genai.GenerativeModel:
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")


def generate_pitch_gemini(company_name: str, description: str) -> str:
    """
    Use Gemini Flash to write exactly 2 sentences explaining why this company
    needs a remote iOS engineer. Falls back to a static template on any error.
    """
    if not GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY not set — using static pitch")
        return STATIC_PITCH_FALLBACK

    try:
        model = _gemini_model()
        prompt = (
            f"You are helping an iOS Engineer write a personalised cold outreach email.\n\n"
            f"Company: {company_name}\n"
            f"What they do: {description[:600]}\n"
            f"Sender background: {SENDER_CONTEXT}\n\n"
            f"Task: Write EXACTLY 2 sentences that explain concretely and specifically "
            f"why {company_name} would benefit from a skilled remote iOS engineer right now. "
            f"Reference their actual product space. Be direct and insightful — not generic. "
            f"Do NOT include a greeting, sign-off, or any extra text. "
            f"Output only the 2 sentences."
        )
        response = model.generate_content(prompt)
        pitch = response.text.strip()

        # Trim to at most 2 sentences
        sentences = re.split(r'(?<=[.!?])\s+', pitch)
        return " ".join(sentences[:2])

    except Exception as exc:
        log.warning(f"Gemini pitch error for '{company_name}': {exc} — using fallback")
        return STATIC_PITCH_FALLBACK


def generate_subject_gemini(company_name: str) -> str:
    """
    Generate a short, compelling subject line. Falls back to a default.
    """
    if not GEMINI_API_KEY:
        return f"iOS Engineer — Interested in {company_name}"

    try:
        model = _gemini_model()
        prompt = (
            f"Write a cold-email subject line (max 9 words) for an iOS Engineer "
            f"reaching out to {company_name} about a remote role. "
            f"Reference their product or sector. Sound human, not salesy. "
            f"Do NOT use 'iOS Developer' verbatim. No quotes. Output only the subject line."
        )
        subject = model.generate_content(prompt).text.strip().strip('"\'')
        # Safety: cap length
        return subject if len(subject) <= 80 else f"iOS Engineer — Interested in {company_name}"

    except Exception as exc:
        log.warning(f"Gemini subject error for '{company_name}': {exc}")
        return f"iOS Engineer — Interested in {company_name}"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Email Composition & Sending
# ══════════════════════════════════════════════════════════════════════════════

def build_email_body(first_name: str, company_name: str, pitch: str) -> str:
    """Compose the full cold-email body."""
    greeting = f"Hi {first_name}," if first_name else "Hi there,"

    return f"""{greeting}

I came across {company_name} and was genuinely impressed by what you're building.

{pitch}

I'm Sharang Verma, an iOS Engineer with 2+ years of experience shipping production apps at scale. Most recently at Pincode (PhonePe), where I worked on a consumer app serving millions of users — optimising performance, managing in-app purchases via StoreKit, and integrating Kotlin Multiplatform for shared business logic. Before that, I built the BNGAI B2B mapping app from the ground up at AiDash, covering all MapKit and geospatial features. I also open-sourced SwiftModelGraph — a code-generation tool built on Swift Macros and IndexStoreDB that's now used internally at scale.

I'm actively looking for a remote iOS role and would love to explore if there's a fit.

I've attached my CV for reference — happy to also share my GitHub or jump on a quick call.

Best,
Sharang Verma
iOS Engineer · Swift · UIKit · SwiftUI · MapKit · KMP
{SENDER_LINKEDIN} | {SENDER_GITHUB}
"""


def send_email(to_email: str, subject: str, body: str, attachment_path: str | None = None) -> bool:
    """
    Send a plain-text email via Gmail SMTP SSL, optionally with a PDF attachment.
    Returns True on success, False on any failure (never raises).
    """
    if not GMAIL_USER or not GMAIL_PASS:
        log.error("GMAIL_USER / GMAIL_PASS not set — cannot send email")
        return False

    try:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"] = f"{SENDER_NAME} <{GMAIL_USER}>"
        msg["To"] = to_email
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # Attach CV if the file exists
        if attachment_path and os.path.isfile(attachment_path):
            with open(attachment_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            filename = os.path.basename(attachment_path)
            part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
            msg.attach(part)
            log.info(f"Attached {filename}")
        elif attachment_path:
            log.warning(f"CV file not found at '{attachment_path}' — sending without attachment")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
            smtp.login(GMAIL_USER, GMAIL_PASS)
            smtp.sendmail(GMAIL_USER, to_email, msg.as_string())

        log.info(f"✓ Email sent → {to_email}")
        return True

    except smtplib.SMTPAuthenticationError:
        log.error(
            "SMTP auth failed. Ensure GMAIL_PASS is a Google App Password "
            "(not your account password) and 2FA is enabled."
        )
        return False
    except Exception as exc:
        log.error(f"Failed to send to {to_email}: {exc}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Orchestration
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("═" * 60)
    log.info("Startup Cold-Email Automation — starting run")
    log.info("═" * 60)

    init_leads_csv(LEADS_CSV)
    contacted_domains = load_contacted_leads(LEADS_CSV)

    # ── 1. Source ──────────────────────────────────────────────────────────────
    yc_companies = fetch_yc_companies(max_pages=5)
    rss_companies = fetch_rss_companies()
    all_companies = yc_companies + rss_companies
    log.info(f"Total sourced: {len(all_companies)} companies")

    # ── 2. Filter ──────────────────────────────────────────────────────────────
    relevant = filter_companies(all_companies)

    # ── 3. Deduplicate (against CSV + within this batch) ──────────────────────
    seen_this_run: set[str] = set()
    new_leads: list[dict] = []

    for company in relevant:
        domain = company.get("domain", "")
        if not domain:
            continue
        if domain in contacted_domains or domain in seen_this_run:
            continue
        seen_this_run.add(domain)
        new_leads.append(company)

    log.info(f"New un-contacted leads: {len(new_leads)}")

    # ── 4. Cap per run ────────────────────────────────────────────────────────
    batch = new_leads[:MAX_NEW_LEADS_PER_RUN]
    log.info(f"Processing batch of {len(batch)} leads this run")

    # ── 5–8. Enrich → Pitch → Send → Save ────────────────────────────────────
    stats = {
        "emailed": 0,
        "emailed_guessed": 0,
        "no_email": 0,
        "email_failed": 0,
    }

    for company in batch:
        name   = company["name"]
        domain = company["domain"]
        desc   = company["description"]

        log.info(f"── {name} ({domain})")

        # Enrich
        apollo_result = find_contact_apollo(domain)
        email, first_name, last_name = resolve_contact_email(domain, apollo_result)

        if not email:
            log.warning(f"No email resolved for {name} — skipping")
            save_lead(LEADS_CSV, domain, name, "", "no_email_found")
            contacted_domains.add(domain)
            stats["no_email"] += 1
            continue

        was_guessed = not (apollo_result and apollo_result.get("email"))

        # Personalize
        pitch   = generate_pitch_gemini(name, desc)
        subject = generate_subject_gemini(name)
        body    = build_email_body(first_name, name, pitch)

        # Send
        success = send_email(email, subject, body, attachment_path=CV_PATH)

        if success:
            status = "emailed_guessed" if was_guessed else "emailed"
            stats["emailed"] += 1
            if was_guessed:
                stats["emailed_guessed"] += 1
        else:
            status = "email_failed"
            stats["email_failed"] += 1

        save_lead(LEADS_CSV, domain, name, email, status)
        contacted_domains.add(domain)

        time.sleep(2)  # polite pause between API calls

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("Run complete:")
    log.info(f"  Sourced:       {len(all_companies)}")
    log.info(f"  Filtered:      {len(relevant)}")
    log.info(f"  New leads:     {len(new_leads)}")
    log.info(f"  Emailed:       {stats['emailed']}"
             f"  (of which {stats['emailed_guessed']} used guessed addresses)")
    log.info(f"  No email:      {stats['no_email']}")
    log.info(f"  Send failed:   {stats['email_failed']}")
    log.info("═" * 60)


if __name__ == "__main__":
    main()
