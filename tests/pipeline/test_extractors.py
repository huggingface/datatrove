import unittest

from datatrove.pipeline.extractors import (
    Inscriptis,
    Justext,
    ReadabiliPy,
    Readability,
    Resiliparse,
    Trafilatura,
)

from ..utils import (
    require_inscriptis,
    require_justext,
    require_readabilipy,
    require_readability,
    require_resiliparse,
    require_trafilatura,
)


ARTICLE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Article</title>
</head>
<body>
    <article>
        <header>
            <h1>The Evolution of Technology</h1>
            <p>By Jane Smith | December 21, 2024</p>
        </header>
        <p>Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.</p>
        <p>One of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.</p>
        <p>As technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.</p>
        <footer>
            <p>&copy; 2024 Tech Innovations Publishing</p>
        </footer>
    </article>
</body>
</html>
"""


class TestExtractors(unittest.TestCase):
    @require_trafilatura
    def test_basic_article_trafilatura(self):
        extractor = Trafilatura()
        self.assertEqual(
            extractor.clean_html(ARTICLE_HTML),
            "<body><p>Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.</p><p>One of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.</p><p>As technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.</p></body>",
        )
        self.assertEqual(
            extractor.extract(ARTICLE_HTML),
            "Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.\nOne of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.\nAs technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.",
        )

    @require_justext
    def test_basic_article_justext(self):
        extractor = Justext()
        self.assertEqual(
            extractor.clean_html(ARTICLE_HTML),
            '<html lang="en">\n\n<body>\n    <article>\n        <header>\n            <h1>The Evolution of Technology</h1>\n            <p>By Jane Smith | December 21, 2024</p>\n        </header>\n        <p>Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.</p>\n        <p>One of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.</p>\n        <p>As technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.</p>\n        <footer>\n            <p>&#169; 2024 Tech Innovations Publishing</p>\n        </footer>\n    </article>\n</body>\n</html>',
        )
        self.assertEqual(
            extractor.extract(ARTICLE_HTML),
            "The Evolution of Technology\n\nTechnology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.\n\nOne of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.\n\nAs technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.",
        )

    @require_resiliparse
    def test_basic_article_resiliparse(self):
        extractor = Resiliparse()
        self.assertEqual(extractor.clean_html(ARTICLE_HTML), ARTICLE_HTML)
        self.assertEqual(
            extractor.extract(ARTICLE_HTML),
            "The Evolution of Technology\n\nBy Jane Smith | December 21, 2024\n\nTechnology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.\n\nOne of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.\n\nAs technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.",
        )

    @require_readabilipy
    def test_basic_article_readabilipy(self):
        extractor = ReadabiliPy()
        self.assertEqual(
            extractor.clean_html(ARTICLE_HTML),
            "<div><title><p>Article</p></title><body><article><header><h1>The Evolution of Technology</h1><p>By Jane Smith | December 21, 2024</p></header><p>Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.</p><p>One of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.</p><p>As technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.</p><footer><p>© 2024 Tech Innovations Publishing</p></footer></article></body></div>",
        )
        self.assertEqual(
            extractor.extract(ARTICLE_HTML),
            "Article\n\nThe Evolution of Technology\n\nBy Jane Smith | December 21, 2024\n\nTechnology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.\n\nOne of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.\n\nAs technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.\n\n© 2024 Tech Innovations Publishing",
        )

    @require_inscriptis
    def test_basic_article_inscriptis_with_trafilatura(self):
        extractor = Inscriptis(preprocessor=Trafilatura())
        self.assertEqual(
            extractor.clean_html(ARTICLE_HTML),
            "<body><p>Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.</p><p>One of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.</p><p>As technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.</p></body>",
        )
        self.assertEqual(
            extractor.extract(ARTICLE_HTML),
            "Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.\n\nOne of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.\n\nAs technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.",
        )

    @require_inscriptis
    @require_justext
    def test_basic_article_inscriptis_with_justext(self):
        extractor = Inscriptis(preprocessor=Justext())
        self.assertEqual(
            extractor.clean_html(ARTICLE_HTML),
            '<html lang="en">\n\n<body>\n    <article>\n        <header>\n            <h1>The Evolution of Technology</h1>\n            <p>By Jane Smith | December 21, 2024</p>\n        </header>\n        <p>Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.</p>\n        <p>One of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.</p>\n        <p>As technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.</p>\n        <footer>\n            <p>&#169; 2024 Tech Innovations Publishing</p>\n        </footer>\n    </article>\n</body>\n</html>',
        )
        self.assertEqual(
            extractor.extract(ARTICLE_HTML),
            "The Evolution of Technology\n\nBy Jane Smith | December 21, 2024\n\nTechnology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.\n\nOne of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.\n\nAs technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.\n\n© 2024 Tech Innovations Publishing",
        )

    @require_inscriptis
    @require_resiliparse
    def test_basic_article_inscriptis_with_resiliparse(self):
        extractor = Inscriptis(preprocessor=Resiliparse())
        self.assertEqual(extractor.clean_html(ARTICLE_HTML), ARTICLE_HTML)
        self.assertEqual(
            extractor.extract(ARTICLE_HTML),
            "The Evolution of Technology\n\nBy Jane Smith | December 21, 2024\n\nTechnology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.\n\nOne of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.\n\nAs technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.\n\n© 2024 Tech Innovations Publishing",
        )

    @require_inscriptis
    @require_readabilipy
    def test_basic_article_inscriptis_with_readabilipy(self):
        extractor = Inscriptis(preprocessor=ReadabiliPy())
        self.assertEqual(
            extractor.clean_html(ARTICLE_HTML),
            "<div><title><p>Article</p></title><body><article><header><h1>The Evolution of Technology</h1><p>By Jane Smith | December 21, 2024</p></header><p>Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.</p><p>One of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.</p><p>As technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.</p><footer><p>© 2024 Tech Innovations Publishing</p></footer></article></body></div>",
        )
        self.assertEqual(
            extractor.extract(ARTICLE_HTML),
            "The Evolution of Technology\n\nBy Jane Smith | December 21, 2024\n\nTechnology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.\n\nOne of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.\n\nAs technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.\n\n© 2024 Tech Innovations Publishing",
        )

    @require_inscriptis
    @require_justext
    @require_readability
    def test_basic_article_inscriptis_with_readability(self):
        extractor = Inscriptis(preprocessor=Readability(postprocessor=Justext()))
        self.assertEqual(
            extractor.clean_html(ARTICLE_HTML),
            '<html><body><div><body id="readabilityBody">\n    <article>\n        \n        <p>Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.</p>\n        <p>One of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.</p>\n        <p>As technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.</p>\n        \n    </article>\n</body>\n</div></body></html>',
        )
        self.assertEqual(
            extractor.extract(ARTICLE_HTML),
            "Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.\n\nOne of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.\n\nAs technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.",
        )

    def test_basic_article_readability_with_trafilatura(self):
        extractor = Readability(postprocessor=Trafilatura())
        self.assertEqual(
            extractor.clean_html(ARTICLE_HTML),
            '<html><body><div><body id="readabilityBody">\n    <article>\n        \n        <p>Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.</p>\n        <p>One of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.</p>\n        <p>As technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.</p>\n        \n    </article>\n</body>\n</div></body></html>',
        )
        self.assertEqual(
            extractor.extract(ARTICLE_HTML),
            "Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.\nOne of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.\nAs technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.",
        )

    def test_basic_article_readability_with_justext(self):
        extractor = Readability(postprocessor=Justext())
        self.assertEqual(
            extractor.clean_html(ARTICLE_HTML),
            '<html><body><div><body id="readabilityBody">\n    <article>\n        \n        <p>Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.</p>\n        <p>One of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.</p>\n        <p>As technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.</p>\n        \n    </article>\n</body>\n</div></body></html>',
        )
        self.assertEqual(
            extractor.extract(ARTICLE_HTML),
            "Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.\n\nOne of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.\n\nAs technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.",
        )

    def test_basic_article_readability_with_resiliparse(self):
        extractor = Readability(postprocessor=Resiliparse())
        self.assertEqual(
            extractor.clean_html(ARTICLE_HTML),
            '<html><body><div><body id="readabilityBody">\n    <article>\n        \n        <p>Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.</p>\n        <p>One of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.</p>\n        <p>As technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.</p>\n        \n    </article>\n</body>\n</div></body></html>',
        )
        self.assertEqual(
            extractor.extract(ARTICLE_HTML),
            "Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.\n\nOne of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.\n\nAs technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.",
        )

    def test_basic_article_readability_with_readabilipy(self):
        extractor = Readability(postprocessor=ReadabiliPy())
        self.assertEqual(
            extractor.clean_html(ARTICLE_HTML),
            '<html><body><div><body id="readabilityBody">\n    <article>\n        \n        <p>Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.</p>\n        <p>One of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.</p>\n        <p>As technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.</p>\n        \n    </article>\n</body>\n</div></body></html>',
        )
        self.assertEqual(
            extractor.extract(ARTICLE_HTML),
            "Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.\n\nOne of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.\n\nAs technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.",
        )

    def test_basic_article_readability_with_inscriptis(self):
        extractor = Readability(postprocessor=Inscriptis(preprocessor=Justext()))
        self.assertEqual(
            extractor.clean_html(ARTICLE_HTML),
            '<html><body><div><body id="readabilityBody">\n    <article>\n        \n        <p>Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.</p>\n        <p>One of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.</p>\n        <p>As technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.</p>\n        \n    </article>\n</body>\n</div></body></html>',
        )
        self.assertEqual(
            extractor.extract(ARTICLE_HTML),
            "Technology has evolved at an unprecedented pace over the past century, revolutionizing industries, societies, and daily life. From the early days of the industrial revolution to the digital age, technological advancements have reshaped the way we live, work, and communicate.\n\nOne of the most significant changes has been the rise of the internet, which has connected the world in ways never before imagined. The advent of personal computers and mobile devices has democratized access to information, transforming industries ranging from healthcare to entertainment.\n\nAs technology continues to advance, the future holds even more promise. Innovations such as artificial intelligence, quantum computing, and renewable energy are poised to redefine our world once again, presenting new opportunities and challenges for generations to come.",
        )
