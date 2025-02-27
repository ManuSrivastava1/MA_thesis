#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/setup.sh"

xp=$1

echo "Preparing files for frame identification..."

rm -fr ${XP_DIR}/${xp}/frameid/data/lexicons
rm -fr ${XP_DIR}/${xp}/frameid/results

mkdir ${XP_DIR}/${xp}/frameid 2> /dev/null
mkdir ${XP_DIR}/${xp}/frameid/data 2> /dev/null
mkdir ${XP_DIR}/${xp}/frameid/data/embeddings 2> /dev/null
mkdir ${XP_DIR}/${xp}/frameid/data/corpora 2> /dev/null
mkdir ${XP_DIR}/${xp}/frameid/data/lexicons 2> /dev/null

cp ${XP_DIR}/${xp}/data/dev.frames ${XP_DIR}/${xp}/frameid/data/corpora/
cp ${XP_DIR}/${xp}/data/dev.sentences.conllx ${XP_DIR}/${xp}/frameid/data/corpora/
cp ${XP_DIR}/${xp}/data/test.frames ${XP_DIR}/${xp}/frameid/data/corpora/
cp ${XP_DIR}/${xp}/data/test.sentences.conllx ${XP_DIR}/${xp}/frameid/data/corpora/
cp ${XP_DIR}/${xp}/data/train.frame.elements ${XP_DIR}/${xp}/frameid/data/corpora/
cp ${XP_DIR}/${xp}/data/train.sentences.conllx.flattened ${XP_DIR}/${xp}/frameid/data/corpora/

#cp ${RESOURCES_DIR}/deps.words.txt ${XP_DIR}/${xp}/frameid/data/embeddings/

mv ${XP_DIR}/${xp}/frameid/data/corpora/dev.frames ${XP_DIR}/${xp}/frameid/data/corpora/dev.frame.elements
mv ${XP_DIR}/${xp}/frameid/data/corpora/test.frames ${XP_DIR}/${xp}/frameid/data/corpora/test.frame.elements

bash ${SCRIPTS_DIR}/flatten.sh -f ${XP_DIR}/${xp}/frameid/data/corpora/dev.sentences.conllx
bash ${SCRIPTS_DIR}/flatten.sh -f ${XP_DIR}/${xp}/frameid/data/corpora/test.sentences.conllx

echo "Done"

