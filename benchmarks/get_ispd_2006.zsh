BENCH_DIR=ispd2006
INS_NAME_LIST=(adaptec5 newblue1 newblue2 newblue3 newblue4 newblue5 newblue6 newblue7)

mkdir $BENCH_DIR
cd $BENCH_DIR
for INS_NAME in $INS_NAME_LIST ; do
    wget https://www.ispd.cc/contests/06/contest/$INS_NAME.tar.gz
    mkdir $INS_NAME
    tar -zxvf $INS_NAME.tar.gz -C $INS_NAME
    cd $INS_NAME
    gunzip *.gz
    cd ..
    rm $INS_NAME.tar.gz
done

cd ..
