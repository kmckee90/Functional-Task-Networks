"""Browser checks on the pod; screenshots stay outside the source sync tree."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

out = Path('/home/dev/proposal_state/qa')
out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    errors = []
    failed = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('response', lambda response: failed.append((response.status, response.url)) if response.status >= 400 else None)
    page.goto('http://127.0.0.1:8080/', wait_until='networkidle')
    page.screenshot(path=str(out/'overview.png'), full_page=True)
    page.goto('http://127.0.0.1:8080/#/projects', wait_until='networkidle')
    assert page.locator('.project-row').count() == 8
    assert page.locator('#nav a[href="#/projects"]').inner_text() == 'Projects'
    for image in page.locator('.project-row img').all():
        image.scroll_into_view_if_needed()
        image.evaluate('(img) => img.decode()')
    for project in page.evaluate('researchProjects.map(p=>p.id)'):
        page.goto('http://127.0.0.1:8080/#/projects/'+project, wait_until='networkidle')
        assert page.locator('main h1').count() == 1
        if project == 'chatftn':
            page.frame_locator('.chat-applet').locator('textarea').first.wait_for(timeout=30000)
        for image in page.locator('main img').all():
            image.scroll_into_view_if_needed()
            image.evaluate('(img) => img.decode()')
    reports = page.evaluate('Object.keys(proposalReports)')
    assert len(reports) == 9
    for name in reports:
        page.goto('http://127.0.0.1:8080/#/notes/'+name, wait_until='networkidle')
        assert page.locator('main h1').count() == 1, name
        assert 'Report preview' not in page.locator('main').inner_text(), name
        for image in page.locator('main img').all():
            image.scroll_into_view_if_needed()
            image.evaluate('(img) => img.decode()')
        for video in page.locator('main video').all():
            duration = video.evaluate('async v => {v.preload="metadata";v.load();await new Promise((ok,bad)=>{v.onloadedmetadata=ok;v.onerror=()=>bad(Error("Video failed"))});return v.duration}')
            assert 0 < duration <= 160, (name, duration)
    page.goto('http://127.0.0.1:8080/#/theory', wait_until='networkidle')
    page.wait_for_selector('.proposal-paper h2')
    assert page.locator('.proposal-inline-link a').count() == 7
    assert page.locator('main a[href^="#/notes/"]').count() == 0
    for href in page.locator('.proposal-inline-link a').evaluate_all('(xs)=>xs.map(x=>x.getAttribute("href"))'):
        assert href.startswith('#/projects/') or href == '#/research-plan', href
    assert '[link:' not in page.locator('main').inner_text()
    assert page.locator('.draft-field').count() == 0
    assert 'EXCLUDE FOR NOW' not in page.locator('main').inner_text()
    assert 'The program will proceed through concrete demonstrations' not in page.locator('main').inner_text()
    assert page.get_by_role('heading', name='Contributions to Technology', exact=True).count() == 1
    page.screenshot(path=str(out/'proposal.png'), full_page=True)
    page.goto('http://127.0.0.1:8080/#/notes/hierarchy', wait_until='networkidle')
    page.screenshot(path=str(out/'report.png'), full_page=True)
    page.set_viewport_size({'width': 390, 'height': 844})
    for route in ('/', '/theory', '/projects', '/projects/diffusion', '/notes/hierarchy'):
        page.goto('http://127.0.0.1:8080/#'+route, wait_until='networkidle')
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), route
    page.goto('http://127.0.0.1:8080/', wait_until='networkidle')
    page.locator('#menu-toggle').click()
    assert page.locator('#nav').is_visible()
    page.locator('#nav a[href="#/theory"]').click()
    page.locator('#nav').wait_for(state='hidden')
    page.goto('http://127.0.0.1:8080/', wait_until='networkidle')
    page.screenshot(path=str(out/'mobile.png'), full_page=True)
    assert not errors, errors
    assert not failed, failed
    result = {'legacy_routes': len(reports), 'proposal_links': 7, 'console_errors': errors,
              'failed_requests': failed, 'mobile_overflow': False, 'montage_max_seconds': 160}
    (out/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    browser.close()
