function generateData(ratingRange, numShops, numElements) {

    //Generate random numbers in the given range, and make a element,
    //which contains them in the number of Shops and ends with the index (0 ~ numShops-1),which has highest number.
    //Add elements as much as numElements counts.
    //One data consists of several elements and each element consists of generated numbers and index.

    const output = [];

    for (let k = 0; k < numElements; k++){
    const element = [];
        let maxIndex = 0;
        for (let i = 0; i < numShops; i++) {
            const seed = Math.round(Math.random() * ratingRange*100)/100;

            element.push(seed);

            if(seed>element[maxIndex]){
                maxIndex = i;
            }
        };

        element.push(maxIndex);
        output.push(element);
    }
    return output;
  }