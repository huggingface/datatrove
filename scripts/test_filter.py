import numpy as np

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.text import PUNCTUATION_SET, split_into_words
from datatrove.utils.typeshelper import Languages

TEXT = """
WebQC.Org logo

Molar Mass, Molecular Weight and Elemental Composition Calculator

Molar mass of CHOCH2OH is 60.0520 g/mol

Formula in Hill system is C2H4O2
Elemental composition of CHOCH2OH:
SymbolElementAtomic weightAtomsMass percent
CCarbon12.0107240.0010 %
HHydrogen1.0079446.7138 %
OOxygen15.9994253.2852 %
Direct link to this page

Please tell about this free chemistry software to your friends!

Computing molar mass (molar weight)

To calculate molar mass of a chemical compound enter it's formula and click 'Calculate!'. In chemical formula you may use:
  • Any chemical element
  • Functional groups: D, Ph, Me, Et, Bu, AcAc, For, Ts, Tos, Bz, TMS, tBu, Bzl, Bn, Dmg
  • parantesis () or brackets [].
  • Common compound names.
Examples of molar mass computations: NaCl, Ca(OH)2, K4[Fe(CN)6], CuSO4*5H2O, water, nitric acid, potassium permanganate, ethanol, fructose.

Computing molecular weight (molecular mass)

To calculate molecular weight of a chemical compound enter it's formula, specify its isotope mass number after each element in square brackets.
Examples of molecular weight computations: C[14]O[16]2, S[34]O[16]2.

Definitions of molecular mass, molecular weight, molar mass and molar weight

  • Molecular mass (molecular weight) is the mass of one molecule of a substance and is expressed in the unified atomic mass units (u). (1 u is equal to 1/12 the mass of one atom of carbon-12)
  • Molar mass (molar weight) is the mass of one mole of a substance and is expressed in g/mol.
Weights of atoms and isotopes are from NIST article.

Give us feedback about your experience with Molecular Weight Calculator.

Related: Molecular weights of amino acids
molecular weights calculated today

Back to Online Chemical Tools Menu
By using this website, you signify your acceptance of Terms and Conditions and Privacy Policy.
© 2015 webqc.org All rights reserved

Have feedback?
Contact us

Choose language
Deutsch
English
Español
Français
Italiano
Nederlands
Polski
Português
Русский
中文
日本語
한국어

How to cite?

WebQC.Org
online education
free homework help
chemistry problems
questions and answers
"""

doc = Document()
doc.text = TEXT


def test_char_dup_ratio(doc, max_non_alpha_words_ratio: float | None = 0.8):
    text = doc.text
    words = split_into_words(text, Languages.english)
    n_words = len(words)

    # that 80 % of words in a document contain at least one alphabetic character
    if (
        max_non_alpha_words_ratio
        # nb of words with at least 1 alpha char < 0.8
        and sum([any((c.isalpha() for c in w)) for w in words]) / n_words < max_non_alpha_words_ratio
    ):
        return False, "gopher_below_alpha_threshold"