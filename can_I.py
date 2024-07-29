import os
from bowlingPins_detector import main
from bowlingPins19 import bowling



main()

bowling()

folder_path = 'C:\\Users\\User\\OneDrive\\桌面\\BowlingPinsDetector\\saved_images'

# 取得資料夾內的所有檔案
files = os.listdir(folder_path)

# 刪除每個檔案
for file in files:
    file_path = os.path.join(folder_path, file)
    os.remove(file_path)
