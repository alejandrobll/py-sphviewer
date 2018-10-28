cd dark_matter
for entry in `ls *.jpg`; do
    convert -thumbnail 500x500! $entry "../../thumbnails_gallery/$entry"
    echo "Converting file $entry"
done
cd ..

cd gas
for entry in `ls *.jpg`; do
    convert -thumbnail 500x500! $entry "../../thumbnails_gallery/$entry"
    echo "Converting file $entry"
done
cd ..

cd stars
for entry in `ls *.jpg`; do
    convert -thumbnail 500x500! $entry "../../thumbnails_gallery/$entry"
    echo "Converting file $entry"
done
cd ..
