"""Render the self-contained exhibition as a paginated PDF on the SiT pod."""
from pathlib import Path
from weasyprint import HTML
import pymupdf as fitz

ROOT = Path(__file__).parent / 'best_results_2026_10_06'


def main():
    source = (ROOT / 'report.html').read_text()
    source = source.replace('<details>', '<div class="all-seeds">').replace('</details>', '</div>')
    source = source.replace('<summary>', '<h4>').replace('</summary>', '</h4>')
    css = '''
      @page { size: A4; margin: 15mm 13mm 17mm;
        @bottom-left { content: "Bayes FTN · Four-image continual recall"; font: 8pt sans-serif; color: #526577; }
        @bottom-right { content: counter(page) " / " counter(pages); font: 8pt sans-serif; color: #526577; }
      }
      body { background: white; padding: 0; margin: 0; max-width: none; font: 10pt/1.4 sans-serif; }
      h1 { font-size: 25pt; } h2 { font-size: 17pt; } h3 { font-size: 13pt; }
      h1,h2,h3,h4 { break-after: avoid; }
      section { break-before: page; padding: 0; margin: 0; }
      .panel { padding: 0; margin: 5mm 0; }
      article { break-inside: avoid; padding-top: 2mm; margin: 4mm 0; }
      article p { margin: 2mm 0; }
      article h4 { margin: 2mm 0; }
      img { max-width: 100%; height: auto; }
      .support { max-width: 100mm; }
      table { font-size: 8pt; width: 100%; table-layout: fixed; }
      th,td { padding: 2mm 1.2mm; overflow-wrap: anywhere; }
      tr { break-inside: avoid; }
      .source { font-size: 7pt; overflow-wrap: anywhere; }
      nav { display: none; }
      .all-seeds { break-inside: avoid; }
      #filters article > img { max-height: 132mm; width: auto; }
    '''
    source = source.replace('</head>', '<style>' + css + '</style></head>')
    pdf = ROOT / 'best_four_image_continual_recall.pdf'
    HTML(string=source, base_url=str(ROOT)).write_pdf(pdf, presentational_hints=True)
    doc = fitz.open(pdf)
    doc.rewrite_images(dpi_threshold=180, dpi_target=160, quality=88, bitonal=False)
    compressed = pdf.with_suffix('.compact.pdf')
    doc.save(compressed, garbage=4, deflate=True)
    doc.close()
    compressed.replace(pdf)
    doc = fitz.open(pdf)
    text = '\n'.join(page.get_text() for page in doc)
    assert len(doc) > 5
    for required in ['Image filters', 'LFW faces', 'Distinct faces', 'Mixed objects', 'No replay', 'Four support images']:
        assert required in text, required
    assert sum(len(page.get_images()) for page in doc) >= 48
    # A compact contact sheet permits a page-by-page visual layout inspection.
    from PIL import Image, ImageOps, ImageDraw
    thumbs = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(.45,.45), alpha=False)
        im = Image.frombytes('RGB', [pix.width,pix.height], pix.samples)
        canvas = Image.new('RGB',(290,415),'#dce3e9')
        canvas.paste(ImageOps.contain(im,(278,390)),(6,6))
        ImageDraw.Draw(canvas).text((10,397),str(i+1),fill='black')
        thumbs.append(canvas)
    sheet=Image.new('RGB',(290*4,415*((len(thumbs)+3)//4)),'white')
    for i,im in enumerate(thumbs):sheet.paste(im,((i%4)*290,(i//4)*415))
    sheet.save(ROOT/'pdf_page_check.jpg')
    print(f'PDF complete: {len(doc)} pages, {pdf.stat().st_size/1e6:.1f} MB; {pdf}')


if __name__ == '__main__':
    main()
