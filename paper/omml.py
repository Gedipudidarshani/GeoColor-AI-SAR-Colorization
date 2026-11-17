"""LaTeX -> MathML (latex2mathml) -> Word OMML (Office's own MML2OMML.XSL), so equations are native, editable
Word equations instead of images."""
import copy
import glob
import os

from latex2mathml.converter import convert
from lxml import etree

_XSL_CANDIDATES = glob.glob(r"C:\Program Files*\Microsoft Office\root\Office1*\MML2OMML.XSL") + \
                  glob.glob(r"C:\Program Files*\Microsoft Office\Office1*\MML2OMML.XSL")
if not _XSL_CANDIDATES:
    raise FileNotFoundError("MML2OMML.XSL not found (it ships with Microsoft Office).")
_XSLT = etree.XSLT(etree.parse(_XSL_CANDIDATES[0]))


def latex_to_omml(latex):
    """Return an <m:oMath> lxml element for the given LaTeX math."""
    mathml = etree.fromstring(convert(latex).encode("utf-8"))
    omml = _XSLT(mathml).getroot()
    return copy.deepcopy(omml)


if __name__ == "__main__":
    el = latex_to_omml(r"L^* = L^*_{pred} + \beta \frac{\sigma(L^*_{pred})}{\sigma(S)} (S - G_\sigma * S)")
    print(etree.QName(el).localname, len(etree.tostring(el)))
