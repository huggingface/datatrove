import unittest

from datatrove.pipeline.formatters.symbol_lines_remover import SymbolLinesFormatter


INPUT_TEXT = """-
-
-
>> Harold Brantley is a 6-2, 283-pound Defensive Tackle from Hershey, PA. He is ranked No. 1385 in the country by 247Sports.


>> Brantley is the No. 45 recruit in Pennsylvania (PA) and is the No. 89 Defensive Tackle in the nation.

>> Brantley has a 247Sports rating of 80, making him a 3-star prospect.
|
|
|
hello ehehe -
|
- oi mate
|

bye"""

FORMATTED_TEXT = """>> Harold Brantley is a 6-2, 283-pound Defensive Tackle from Hershey, PA. He is ranked No. 1385 in the country by 247Sports.


>> Brantley is the No. 45 recruit in Pennsylvania (PA) and is the No. 89 Defensive Tackle in the nation.

>> Brantley has a 247Sports rating of 80, making him a 3-star prospect.
hello ehehe -
- oi mate

bye"""


class TestSymbolLinesRemover(unittest.TestCase):
    def test_symbol_lines_remover(self):
        formatter = SymbolLinesFormatter()
        self.assertEqual(formatter.format(INPUT_TEXT), FORMATTED_TEXT)
