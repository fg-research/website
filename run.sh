rm -r docs
make clean
make html
git add --all
git commit -m "update website"
git push origin master