#!/bin/sh
val=`sacct --cluster wice -o JobID`
echo $val
stringarray=($val)
for i in "${stringarray[@]}"
do
  echo $i
  `scancel $i --cluster wice`
done

echo 'Done!!'