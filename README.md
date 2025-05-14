# Asessment by Digantara
Detection and Classification of the streaks and stars are satelite images
![preprocessed-image](./src/preprocessed_image.png)

[Problem PDF](./src/problem.pdf)


### Unsigned int16 to Unsigned int8 conversion
Advantages :
- this conversion saves computation cost for better efficient model deployment
- most of the tools assume pixel to be of 8 bits; better compatibility 
- the noise with little dots and line (prolly stars and streak too far) are removed providing more robust data for the classification

Disadvantages :  
- lost of data; the differentiating boundary between the star and streak which were much further away and the ones up close has become less clearer