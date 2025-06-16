# Load and parse the dataset metadata
$json = Get-Content -Raw -Path "dataset_metadata.json" | ConvertFrom-Json

# Base download URL for Harvard Dataverse
$base_url = "https://dataverse.harvard.edu/api/access/datafile"

# Loop through each file entry
foreach ($item in $json.data.latestVersion.files) {
    $file_id = $item.dataFile.id
    $filename = $item.dataFile.filename
    $folder = $item.directoryLabel

    # Compute output directory and create it (recursively, if needed)
    if ($folder) {
        $output_dir = Join-Path -Path "." -ChildPath $folder
    } else {
        $output_dir = "."
    }

    if (-not (Test-Path $output_dir)) {
        New-Item -ItemType Directory -Path $output_dir -Force | Out-Null
    }

    $output_path = Join-Path -Path $output_dir -ChildPath $filename

    # Download the file
    Write-Host "Downloading: $filename → $output_path"
    Invoke-WebRequest -Uri "$base_url/$file_id" -OutFile $output_path
}
