import smtplib
from email.mime.text import MIMEText
from config.settings import settings


def send_email(subject: str, body: str, to: str | None = None) -> None:
    if not settings.smtp_host:
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = settings.alert_email_from
    msg["To"] = to or settings.alert_email_from

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as smtp:
        smtp.starttls()
        smtp.login(settings.smtp_user, settings.smtp_password)
        smtp.send_message(msg)
