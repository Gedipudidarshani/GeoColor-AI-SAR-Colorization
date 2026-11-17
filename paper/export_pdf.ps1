# Export the Word paper to PDF with Microsoft Word (embeds fonts, as IEEE PDF eXpress requires).
param([string]$Docx = "$PSScriptRoot\GeoColor_IEEE_Conference_Paper_final.docx")
$pdf = [System.IO.Path]::ChangeExtension($Docx, ".pdf")
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $doc = $word.Documents.Open($Docx, $false, $true)
    $pages = $doc.ComputeStatistics(2)   # wdStatisticPages
    $doc.ExportAsFixedFormat($pdf, 17)   # wdExportFormatPDF
    $doc.Close($false)
    Write-Output "pages=$pages"
    Write-Output "pdf=$pdf"
} finally {
    $word.Quit()
}
