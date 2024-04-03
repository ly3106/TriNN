grep -RlE "Pedestrian|Cyclist" ./ | xargs -I {} basename {} | sed 's/\.[^.]*$//' | sort > ~/_pedestrian_cyclist.txt
