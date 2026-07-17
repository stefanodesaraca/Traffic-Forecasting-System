param(
    [int]$TimeoutSeconds = 300
)

Write-Host "Starting E2E environment using docker-compose.e2e.yml"

docker compose -f docker-compose.e2e.yml up -d

$start = Get-Date

function Wait-Port($host, $port, $timeoutSec) {
    $end = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $end) {
        try {
            $res = Test-NetConnection -ComputerName $host -Port $port -WarningAction SilentlyContinue
            if ($res.TcpTestSucceeded) { return $true }
        } catch {}
        Start-Sleep -Seconds 1
    }
    return $false
}

Write-Host "Waiting for RabbitMQ (localhost:5672)"
if (-not (Wait-Port -host "localhost" -port 5672 -timeoutSec 120)) {
    Write-Error "RabbitMQ did not become ready in time"
    exit 1
}

Write-Host "Waiting for Postgres (localhost:5432)"
if (-not (Wait-Port -host "localhost" -port 5432 -timeoutSec 120)) {
    Write-Error "Postgres did not become ready in time"
    exit 1
}

Write-Host "Running E2E tests"
$env:ENABLE_E2E = "1"
$env:POSTGRES_HOST = "localhost"
$env:RABBITMQ_HOST = "localhost"

python -m unittest discover -s services/integration_tests -p "test_e2e_compose.py" -v
$exitCode = $LASTEXITCODE

Write-Host "Tearing down Docker Compose"
docker compose -f docker-compose.e2e.yml down

exit $exitCode
