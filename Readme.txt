nvidia/nemotron-3-nano-30b-a3b:free

# Tìm PID đang dùng port 8000
netstat -ano | findstr ":8000"

# Kill theo PID (thay 7644 bằng PID thực tế)
Stop-Process -Id 7644 -Force
