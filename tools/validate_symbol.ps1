param(
  [Parameter(Mandatory=$true)][string]$Symbol,
  [string]$Tf = '1h'
)
$ErrorActionPreference = 'Stop'
$uri = "http://127.0.0.1:5000/api/analyze/$Symbol?diag=1&validate=1&tf=$Tf"
Write-Host "GET $uri" -ForegroundColor Cyan
$r = Invoke-RestMethod -Method GET -Uri $uri -TimeoutSec 120
if(-not $r.success){ throw "Request failed: $($r.error)" }
$r.data.final_score.validation | ConvertTo-Json -Depth 6
if($r.data.trade_setups){
  $r.data.trade_setups[0] | Select-Object direction, strategy, entry, stop_loss, confidence, primary_rr, @{n='t1';e={$_.targets[0].price}} | Format-List
}
