import unittest

from datatrove.pipeline.formatters import PIIFormatter


IP_TEST_INPUT = """It correctly doesn't match this invalid ips:
999.999.999.999.
108.0.0.01
0.1.2.3
00.0000.00.00
192.168.l.1
912.456.123.123
.3.3.3.0
192.168.o.0

It doesn't match local IPs:

172.16.0.9
172.16.4.1
172.17.1.1
127.0.0.2
10.0.1.5
10.0.0.1
10.155.155.155
10.255.255.254
172.16.0.4
172.16.0.1
172.17.1.1
172.31.254.254
192.168.1.2
192.168.254.0

valid ips:
163.36.107.25
114.35.103.54
57.120.21.121
13.83.69.70
29.116.31.158
193.163.229.23
211.160.67.46
4.132.211.142
214.70.227.4
35.88.121.75"""

IP_TEST_OUTPUT = """It correctly doesn't match this invalid ips:
999.999.999.999.
108.0.0.01
0.1.2.3
00.0000.00.00
192.168.l.1
912.456.123.123
.IP
192.168.o.0

It doesn't match local IPs:

172.16.0.9
172.16.4.1
172.17.1.1
127.0.0.2
10.0.1.5
10.0.0.1
10.155.155.155
10.255.255.254
172.16.0.4
172.16.0.1
172.17.1.1
172.31.254.254
192.168.1.2
192.168.254.0

valid ips:
IP
IP
IP
IP
IP
IP
IP
IP
IP
IP"""


EMAIL_TEST_INPUT = """Use: for testing against email regex
ref: http://codefool.tumblr.com/post/15288874550/list-of-valid-and-invalid-email-addresses


List of Valid Email Addresses

email@example.com
firstname.lastname@example.com
email@subdomain.example.com
firstname+lastname@example.com
email@123.123.123.123
email@[123.123.123.123]
"email"@example.com
1234567890@example.com
email@example-one.com
_______@example.com
email@example.name
email@example.museum
email@example.co.jp
firstname-lastname@example.com
NAME@MYSITE.COM



List of Strange Valid Email Addresses

much.”more\\ unusual”@example.com
very.unusual.”@”.unusual.com@example.com
very.”(),:;<>[]”.VERY.”very@\\ "very”.unusual@strange.example.com



List of Invalid Email Addresses

plainaddress
#@%^%#$@#$@#.com
@example.com
Joe Smith <email@example.com>
email.example.com
email@example@example.com
.email@example.com
email.@example.com
email..email@example.com
あいうえお@example.com
email@example.com (Joe Smith)
email@example
email@-example.com
email@example.web
email@111.222.333.44444
email@example..com
Abc..123@example.com



List of Strange Invalid Email Addresses

”(),:;<>[\\]@example.com
just”not”right@example.com
this\\ is"really"not\\allowed@example.com"""

EMAIL_TEST_OUTPUT = r"""Use: for testing against email regex
ref: http://codefool.tumblr.com/post/15288874550/list-of-valid-and-invalid-email-addresses


List of Valid Email Addresses

EMAIL
EMAIL
EMAIL
EMAIL
EMAIL
EMAIL
"email"@example.com
EMAIL
EMAIL
EMAIL
EMAIL
EMAIL
EMAIL
EMAIL
EMAIL



List of Strange Valid Email Addresses

much.”more\ unusual”@example.com
very.unusual.”@”.EMAIL
very.”(),:;<>[]”.VERY.”very@\ "very”.EMAIL



List of Invalid Email Addresses

plainaddress
#@%^%#$@#$@#.com
@example.com
Joe Smith <EMAIL>
email.example.com
email@EMAIL
.EMAIL
email.@example.com
email..EMAIL
あいうえお@example.com
EMAIL (Joe Smith)
email@example
email@-example.com
EMAIL
EMAIL
email@example..com
Abc..EMAIL



List of Strange Invalid Email Addresses

”(),:;<>[\]@example.com
just”not”EMAIL
this\ is"really"not\EMAIL"""


class TestPIIRemoval(unittest.TestCase):
    def test_pii_removal(self):
        remover = PIIFormatter(
            email_replacement="EMAIL",
            ip_replacement="IP",
        )
        self.assertEqual(remover.format(IP_TEST_INPUT), IP_TEST_OUTPUT)
        self.assertEqual(remover.format(EMAIL_TEST_INPUT), EMAIL_TEST_OUTPUT)
