rm -r docs
make html
#sphobjinv suggest docs/objects.inv as_rst -st 0
git add --all
git commit -m "update website"
git push origin master