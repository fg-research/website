rm -r docs
make html
cp ads.txt docs/ads.txt
git add --all
git commit -m "update website"
git push origin master