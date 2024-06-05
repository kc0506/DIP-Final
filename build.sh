folder="Team21_FinalPackage"
mkdir $folder
cp ./report/main.pdf ./$folder/
cp -r ./project ./$folder/
cp ./slides.pdf ./$folder/
zip -r $folder.zip $folder
rm -r $folder