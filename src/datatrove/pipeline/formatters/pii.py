import ipaddress
import re
from functools import partial
from typing import Callable

from datatrove.pipeline.formatters.base import BaseFormatter


class PIIReplacer:
    def __init__(
        self, regex: str, replacements: tuple[str, ...] | str, validator: Callable[[str], bool] | None = None
    ):
        self.regex: re.Pattern = re.compile(regex)
        self.replacements = (
            replacements
            if type(replacements) is tuple
            else (tuple(replacements) if not isinstance(replacements, str) else (replacements,))
        )
        self.validator = validator  # extra validation for a match
        self._replace_i = 0

    def replace(self, text: str):
        def get_replacement(matchobj):
            if self.validator and not self.validator(matchobj.group(0)):
                # not a valid match. replace with itself
                return matchobj.group(0)
            replacement = self.replacements[self._replace_i]
            self._replace_i = (self._replace_i + 1) % len(self.replacements)
            return replacement

        return self.regex.sub(get_replacement, text)


def public_ip_validator(ip, public_only: bool = True) -> bool:
    try:
        ip = ipaddress.ip_address(ip)
        return not public_only or ip.is_global
    except ValueError:
        return False


class PIIFormatter(BaseFormatter):
    """
    Replaces email addresses and ip addresses in the document text.
    Args:
        remove_emails: Replace email addresses
        remove_ips: Replace IP addresses
        only_remove_public_ips: by default we only replace public (and thus PII) IPs
        email_replacement: tuple of strings to use as replacement. They will be used in a circular way
        ip_replacement same as email_replacement but for IP addresses
    """

    name = "📞 PII"

    def __init__(
        self,
        remove_emails: bool = True,
        remove_ips: bool = True,
        only_remove_public_ips: bool = True,
        # example.com/org are actually maintained as an example
        email_replacement: tuple[str, ...] | str = ("email@example.com", "firstname.lastname@example.org"),
        # randomly generated list of ips. they did not respond to ping requests at the time the list was created
        ip_replacement: tuple[str, ...] | str = (
            "22.214.171.124",
            "126.96.36.199",
            "188.8.131.52",
            "184.108.40.206",
            "220.127.116.11",
            "18.104.22.168",
        ),
    ):
        super().__init__()
        self.remove_emails = remove_emails
        self.remove_ips = remove_ips

        self.emails_replacer = PIIReplacer(
            r"\b[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:(?:[A-Za-z0-9](?:["
            r"A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|["
            r"01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[A-Za-z0-9-]*[A-Za-z0-9]:)])",
            email_replacement,
        )

        self.ip_replacer = PIIReplacer(
            r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
            validator=partial(public_ip_validator, public_only=only_remove_public_ips),
            replacements=ip_replacement,
        )

    def format(self, text: str) -> str:
        if self.remove_emails:
            text = self.emails_replacer.replace(text)
        if self.remove_ips:
            text = self.ip_replacer.replace(text)
        return text
