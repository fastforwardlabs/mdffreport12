let puppeteer = require('puppeteer')
let fs = require('fs-extra')
let path = require('path')

let margin = '0.5in'

;(async () => {
  let browser = await puppeteer.launch()
  let page = await browser.newPage()
  await page.goto(`file:${path.join(__dirname, 'out/index.html')}`, {
    waitUntil: 'networkidle2',
  })

  await page.addStyleTag({
    content:
      'html { font-size: 12px; line-height: 18px; } body { padding-left: 0; } .table-of-contents { display: none; } figcaption { font-size: 10px; line-height: 18px; }',
  })
  await page.pdf({
    path: 'out/ff12-deep-learning-for-anomaly-detection.pdf',
    height: '8.5in',
    width: '5.5in',
    displayHeaderFooter: true,
    margin: {
      top: margin,
      left: margin,
      right: margin,
      bottom: margin,
    },
  })
  await browser.close()
})()
