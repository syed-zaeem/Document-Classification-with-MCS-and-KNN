from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


from docx import Document
def truncate_text(text, num_words):
    words = text.split()
    truncated_words = words[:num_words]
    truncated_text = ' '.join(truncated_words)
    return truncated_text


driver = webdriver.Chrome()
driver.get("https://www.mydomaine.com/")
elements = driver.find_elements(By.CSS_SELECTOR,"a[class^='comp card']")
hrefs = []
for i in elements:
    hrefs.append(i.get_attribute('href'))

for index in range(0,15):
    doc = Document()
    driver.get(hrefs[index])
    WebDriverWait(driver,30).until(EC.presence_of_all_elements_located((By.XPATH,'//article')))
    article = driver.find_element(By.XPATH,'//article').text
    truncated_text = truncate_text(article, 500)

    doc.add_paragraph(truncated_text)
    doc.save("scrapedDocuments/"+"Document-"+str(index+1)+"_LifestyleAndHobbies.docx")


