PWD=`pwd`
echo $PWD
activate () {
    . $PWD/temp/bin/activate
}
activate
rm -rf ScProject
git clone https://github.com/parvathysajee/ScProject.git
cd ScProject/sample-code 
python classify.py  --model-name test.h5 --captcha-dir input_data --output stuff.txt --symbols symbols.txt