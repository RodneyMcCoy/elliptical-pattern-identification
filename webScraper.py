from requests_html import HTMLSession

session = HTMLSession()

r = session.get('http://www.yourjspage.com')

r.html.render()  # this call executes the js in the page

