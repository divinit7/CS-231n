using Images, DataFrames, CSV

function read_data(typeData, labelsInfo, imageSize, path)
    #Initialize x matrix
    x = zeros(size(labelsInfo, 1), imageSize)
    for (index, idImage) in enumerate(labelsInfo)
        #read image file
        nameFile = "$(path)/$(typeData)Resized/$(idImage).Bmp"
        img = load(nameFile)

        #Convert img to float values
        temp = channelview( img)
        temp = map(Float64,temp)
        #Convert color images to gray images
        #By taking the average of the color scales
        # if ndims(temp) == 3
        temp = sum.(temp) / 3
        # end

        #Transform image matrix to a vector and store
        #it in data matrix
        x[index, :] = reshape(temp, 1, imageSize)
    end
    return x
end
path = "C:\\Users\\chauh\\Documents\\Github\\CS-231n\\CS231n - in julia\\knn"
ImgSize = 1200 #20 * 20
labelsInfoTrain, data  = eachcol(CSV.File("$(path)/trainLabels.csv") |> DataFrame)
  # = df[:,1], df[:,2]
xTrain = read_data("train" , labelsInfoTrain, ImgSize, path)
labelsInfoTest, testdata = eachcol(CSV.File("$(path)/sampleSubmission.csv") |> DataFrame)
xTest = read_data("test" , labelsInfoTest, ImgSize, path)

yTrain = map(x -> x[1], labelsInfoTrain)

# Convert from character to Integer
yTrain = Int.(yTrain)

using DecisionTree
model = build_forest(yTrain, xTrain, 20, 50, 1.0)
predTest = apply_forest(model, xTest)

labelsInfoTest = Char(predTest[])
CSV.write("$(path)/juliaSubmission.csv", labelsInfoTest, seperator = ',', header = true)
