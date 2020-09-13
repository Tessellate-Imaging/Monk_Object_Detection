set -e

# download image data
bash gdrivedl.sh https://drive.google.com/file/d/1bC68CzsSVTusZVvOkk7imSZSbgD1MqK2/view?usp=sharing totaltext.zip

unzip totaltext.zip
chmod -R o-w Images
rm -rf __MACOSX
mv Images/Train/img61.JPG Images/Train/img61.jpg

# download ground truth data
bash gdrivedl.sh https://drive.google.com/file/d/19quCaJGePvTc3yPZ7MAGNijjKfy77-ke/view?usp=sharing groundtruth_text.zip

unzip groundtruth_text.zip -d gt
chmod -R o-w gt
rm -rf gt/__MACOSX
mv gt/Groundtruth/Polygon/* gt/
rm -rf gt/Groundtruth

mkdir total-text
mv Images gt total-text

mv total-text ../../data
 
