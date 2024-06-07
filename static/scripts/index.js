function uploadImage() {
    document.getElementById('image-upload-form').submit();
}


document.addEventListener('DOMContentLoaded', function () {
    var scale = 1;

    changeMainImage();
    document.querySelector('#resetButton').addEventListener('click', function () {
        document.querySelector('#main-image').src = current_image_url;
        document.getElementById('featuresDiv').innerHTML = '';
    });

    // Zoom feature
    const container = document.getElementById("image-container");
    const img = document.getElementById("main-image");

    container.addEventListener("mousemove", (e) => {
        const x = e.clientX - e.target.offsetLeft;
        const y = e.clientY - e.target.offsetTop;


        img.style.transformOrigin = x + 'px ' + y + 'px';
        img.style.transform = 'scale(' + scale + ')';
    })

    container.addEventListener("mouseleave", () => {
        img.style.transition = "transform 0.2s";
        img.style.transform = "scale(1)";
    })

    container.addEventListener("wheel", (e) => {
        e.preventDefault(); // Prevent default scroll behavior

        scale += e.deltaY * -0.01; // Adjust scale based on scroll direction
        scale = Math.max(1, scale); // Ensure minimum scale is 2

        const x = e.clientX - container.offsetLeft;
        const y = e.clientY - container.offsetTop;

        img.style.transformOrigin = x + 'px ' + y + 'px';
        img.style.transform = 'scale(' + scale + ')';
    });

});


// Function to change the main image
function changeMainImage() {
    // Get the value of the 'img' URL variable
    const urlParams = new URLSearchParams(window.location.search);
    const imgUrl = urlParams.get('img');

    // If 'img' URL variable is present, update the image source
    if (imgUrl) {
        const mainImage = document.querySelector('.container-div img');
        current_image_url = static_images_url + imgUrl;
        mainImage.src = current_image_url
    }
}

// Event listener for 'popstate' event
window.addEventListener('popstate', function () {
    changeMainImage();
});


document.addEventListener("DOMContentLoaded", function () {
    function fillFeaturesDiv(text, isHtmlText = false) {
        if (isHtmlText) {
            document.getElementById('featuresDiv').innerHTML += text;
        } else {
            document.getElementById('featuresDiv').innerHTML += '<div><p class="fw-bold d-inline">' + text[0] + '</p> ' + text[1] + '</div>'
        }
    }

    document.querySelector('#huMomentsGrayButton').addEventListener('click', function () {
        // Get the URL of the uploaded image
        var imageUrl = document.querySelector('#main-image').src

        // get the substring after the last slash
        imageUrl = imageUrl.substring(imageUrl.lastIndexOf('/') + 1);
        var csrftoken = getCookie('csrftoken');

        // Make an AJAX request to the Django endpoint to convert the image to grayscale
        fetch(huMoments, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken
            },
            body: JSON.stringify({image_url: imageUrl, type: 'gray'})
        })
            .then(response => response.json())
            .then(data => {
                // Once the response is received, update the image source with the grayscale image
                // document.querySelector('#main-image').src = data.grayscale_image_path;
                document.querySelector('#main-image').src = data.binary_image_path;
                document.getElementById('featuresDiv').innerHTML = '';
                // print the type of the data.hu_moments[0] element
                for (let i = 0; i < data.hu_moments.length; i++) {
                    fillFeaturesDiv(['Hu Moment ' + i + ': ', data.hu_moments[i]]);
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
    });

    document.querySelector('#huMomentsColorButton').addEventListener('click', function () {
        // Get the URL from the image selected in the URL
        var imageUrl = current_image_url;

        // get the substring after the last slash
        imageUrl = imageUrl.substring(imageUrl.lastIndexOf('/') + 1);
        var csrftoken = getCookie('csrftoken');

        // Make an AJAX request to the Django endpoint to convert the image to grayscale
        fetch(huMoments, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken
            },
            body: JSON.stringify({image_url: imageUrl, type: 'color'})
        })
            .then(response => response.json())
            .then(data => {
                document.getElementById('featuresDiv').innerHTML = '';
                // print the type of the data.hu_moments[0] element
                for (let i = 0; i < data.hu_moments_b.length; i++) {
                    fillFeaturesDiv(['Hu Moment B' + i + ': ', data.hu_moments_b[i]]);
                }
                for (let i = 0; i < data.hu_moments_g.length; i++) {
                    fillFeaturesDiv(['Hu Moment G' + i + ': ', data.hu_moments_g[i]]);
                }
                for (let i = 0; i < data.hu_moments_r.length; i++) {
                    fillFeaturesDiv(['Hu Moment R' + i + ': ', data.hu_moments_r[i]]);
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
    });


    document.querySelector('#grayScaleButton').addEventListener('click', function () {
        // Get the URL of the uploaded image
        var imageUrl = document.querySelector('#main-image').src

        // get the substring after the last slash
        imageUrl = imageUrl.substring(imageUrl.lastIndexOf('/') + 1);


        // Get the CSRF token from the cookie
        var csrftoken = getCookie('csrftoken');

        // Make an AJAX request to the Django endpoint to convert the image to grayscale
        fetch(convertToGrayscaleURL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken
            },
            body: JSON.stringify({image_url: imageUrl})
        })
            .then(response => response.json())
            .then(data => {
                // Once the response is received, update the image source with the grayscale image
                document.querySelector('#main-image').src = data.grayscale_image_path;
            })
            .catch(error => {
                console.error('Error:', error);
            });
    });

    function generateHaralickTable(property, data){
        html = `
            <table class="table table-bordered table-striped table-hover table-sm">
                <thead>
                    <tr>
                        <th scope="col">Distance/Angle</th>
                        ${data.values[0].values.map((value, index) => `<th scope="col">${Math.round(radiansToDegrees(value.angle))}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${data.values.map((value, index) => `
                        <tr>
                            <th scope="row">${value.distance}</th>
                            ${value.values.map((val, index) => `<td>${parseFloat(val.value.toFixed(4))}</td>`).join('')}
                        </tr>
                    `).join('')}
                </tbody>
                
            </table>
               `;
        return html;
    }

    function radiansToDegrees(radians) {
        return radians * (180 / Math.PI);
    }

    function searchProperty(data, property){
        for (let i = 0; i < data.length; i++) {
            if (data[i].property === property) {
                return data[i];
            }
        }
    }

    document.querySelector('#haralickButton').addEventListener('click', function () {
        // Get the URL of the uploaded image
        var imageUrl = document.querySelector('#main-image').src

        // get the substring after the last slash
        imageUrl = imageUrl.substring(imageUrl.lastIndexOf('/') + 1);


        // Get the CSRF token from cthe cookie
        var csrftoken = getCookie('csrftoken');

        // Make an AJAX request to the Django endpoint to convert the image to grayscale
        fetch(generateHaralickFeaturesURL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken
            },
            body: JSON.stringify({image_url: imageUrl})
        })
            .then(response => response.json())
            .then(data => {
                // Once the response is received, update the image source with the grayscale image
                console.log(data)
                // document.querySelector('#main-image').src = data.img_path;
                // Search for the position with the property value as contrast in the data object

                console.log(searchProperty(data, 'contrast'));
//'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation',
//                   'ASM'
                document.getElementById('featuresDiv').innerHTML = '';
                buttons_html = `
                <div class="mb-2 h-100">
                    <div class="btn-group-horizontal h-100" role="group" aria-label="Vertical radio toggle button group">
                      <input type="radio" class="btn-check hara_select" name="vbtn-radio" id="vbtn-contrast" autocomplete="off" checked>
                      <label class="btn btn-outline-danger" for="vbtn-contrast">Contrast</label>
                      <input type="radio" class="btn-check hara_select" name="vbtn-radio" id="vbtn-dissimilarity" autocomplete="off">
                      <label class="btn btn-outline-danger" for="vbtn-dissimilarity">Dissimilarity</label>
                      <input type="radio" class="btn-check hara_select" name="vbtn-radio" id="vbtn-homogeneity" autocomplete="off">
                      <label class="btn btn-outline-danger" for="vbtn-homogeneity">Homogeneity</label>
                      <input type="radio" class="btn-check hara_select" name="vbtn-radio" id="vbtn-energy" autocomplete="off">
                      <label class="btn btn-outline-danger" for="vbtn-energy">Energy</label>
                      <input type="radio" class="btn-check hara_select" name="vbtn-radio" id="vbtn-correlation" autocomplete="off">
                      <label class="btn btn-outline-danger" for="vbtn-correlation">Correlation</label>
                      <input type="radio" class="btn-check hara_select" name="vbtn-radio" id="vbtn-asm" autocomplete="off">
                      <label class="btn btn-outline-danger" for="vbtn-asm">ASM</label>
                    </div>
                </div>
                <div id="haralickTable">
                </div>
                `;
                document.getElementById('featuresDiv').innerHTML += buttons_html;

                document.getElementById('haralickTable').innerHTML = generateHaralickTable('Contrast', searchProperty(data, 'contrast'));

                document.querySelectorAll('.hara_select').forEach(radio => {
                    radio.addEventListener('change', function () {
                        if (this.checked) {
                            const selectedButton = this.id.replace('vbtn-', '');
                            document.getElementById('haralickTable').innerHTML = generateHaralickTable(selectedButton, searchProperty(data, selectedButton));
                        }
                    });
                });


                // document.getElementById('featuresDiv').innerHTML += generateHaralickTable('Contrast', searchProperty(data, 'contrast'));

                // fillFeaturesDiv(['Contrast: ', data.contrast]);
                // fillFeaturesDiv(['Dissimilarity: ', data.dissimilarity]);
                // fillFeaturesDiv(['Homogeneity: ', data.homogeneity]);
                // fillFeaturesDiv(['Energy: ', data.energy]);
                // fillFeaturesDiv(['Correlation: ', data.correlation]);
            })
            .catch(error => {
                console.error('Error:', error);
            });
    });

    document.querySelector('#xgboost-binary').addEventListener('click', function () {
        // Get the URL of the uploaded image
        var imageUrl = document.querySelector('#main-image').src

        // get the substring after the last slash
        imageUrl = imageUrl.substring(imageUrl.lastIndexOf('/') + 1);
        var csrftoken = getCookie('csrftoken');

        // Make an AJAX request to the Django endpoint to convert the image to grayscale
        fetch(classify, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken
            },
            body: JSON.stringify({image_url: imageUrl, model: 'xgboostBinary'})
        })
            .then(response => response.json())
            .then(data => {
                let clazz = parseInt(data.img_class);
                if (clazz == 0) {
                    document.getElementById('featuresDiv').innerHTML = '';
                    fillFeaturesDiv(['Class:', 'Negative for intraepithelial lesion']);
                } else {
                    document.getElementById('featuresDiv').innerHTML = '';
                    fillFeaturesDiv(['Class:', 'Positive for intraepithelial lesion']);
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
    });

    document.querySelector('#xgboost-multiclass').addEventListener('click', function () {
        // Get the URL of the uploaded image
        var imageUrl = document.querySelector('#main-image').src

        // get the substring after the last slash
        imageUrl = imageUrl.substring(imageUrl.lastIndexOf('/') + 1);
        var csrftoken = getCookie('csrftoken');

        // Make an AJAX request to the Django endpoint to convert the image to grayscale
        fetch(classify, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken
            },
            body: JSON.stringify({image_url: imageUrl, model: 'xgboostMulticlass'})
        })
            .then(response => response.json())
            .then(data => {
                document.getElementById('featuresDiv').innerHTML = '';
                fillFeaturesDiv(['Class:', data['img_class']]);
            })
            .catch(error => {
                console.error('Error:', error);
            });
    });

});

// Function to get the CSRF token from the cookie
function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function initChartDiv() {
    document.getElementById('featuresDiv').innerHTML =
        '<div className="graph-container text-center w-100" style="width: 400px !important;" id="chartDiv"> <canvas className="text-center w-100" id="myChart"style="max-width: 1000px; height: 500px !important; width: 400px !important;"></canvas> </div> <div className="text-center align-content-center w-100 mt-1" style="width: 400px !important;" id="highestValues"> </div>';
}

document.addEventListener("DOMContentLoaded", function () {
    document.querySelector('#histogramButton').addEventListener('click', function () {
        initChartDiv();
        // Get the URL of the uploaded image
        var imageUrl = document.querySelector('#main-image').src;

        // Get the substring after the last slash
        imageUrl = imageUrl.substring(imageUrl.lastIndexOf('/') + 1);

        // Get the CSRF token from the cookie
        var csrftoken = getCookie('csrftoken');

        // Make an AJAX request to the Django endpoint to generate the histogram data
        fetch(generateHistogram, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken
            },
            body: JSON.stringify({image_url: imageUrl})
        })
            .then(response => response.json())
            .then(data => {
                console.log(data);

                // Once the response is received, create the histogram chart
                if (data.imgType === 'grayscale') {
                    createHistogramChartGrayScale(data.histogram);
                } else {
                    createHistogramChartHSV(data.histogram_h, data.histogram_s, data.histogram_v);
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
    });
});

function createHistogramChartHSV(histogramH, histogramS, histogramV) {
    initChartDiv();
    // Define labels for histogram bins (intensity levels)
    var labels = Array.from(Array(256).keys());

    // Create the dataset
    var datasetH = {
        label: 'Hue',
        backgroundColor: 'rgba(255, 99, 132)',
        borderColor: 'rgb(255, 99, 132)',
        data: histogramH
    };

    var datasetS = {
        label: 'Saturation',
        backgroundColor: 'rgba(54, 162, 235, 1)',
        borderColor: 'rgb(54, 162, 235)',
        data: histogramS
    };

    var datasetV = {
        label: 'Value',
        backgroundColor: 'rgba(255, 206, 86, 1)',
        borderColor: 'rgb(255, 206, 86)',
        data: histogramV
    };

    // Configuration options for the chart
    var options = {
        scales: {
            x: {
                beginAtZero: true // Ensure the x-axis starts at zero
            },
            y: {
                beginAtZero: true // Ensure the y-axis starts at zero
            }
        }
    };

    // Create the data object
    var data = {
        labels: labels,
        datasets: [datasetH, datasetS, datasetV]
    };

    // Get the canvas element
    var ctx = document.getElementById('myChart').getContext('2d');

    // Create the chart
    var histogramChart = new Chart(ctx, {
        type: 'bar',
        data: data,
        options: options
    });
    let indexMaxValueH = histogramH.indexOf(Math.max(...histogramH));
    let indexMaxValueS = histogramS.indexOf(Math.max(...histogramS));
    let indexMaxValueV = histogramV.indexOf(Math.max(...histogramV));
    document.getElementById('highestValues').innerHTML = '<div><p class="fw-bold d-inline"> Highest amount of Hue:</p> (' + indexMaxValueH + ', ' + histogramH[indexMaxValueH] + ')</div>';
    document.getElementById('highestValues').innerHTML += '<div><p class="fw-bold d-inline"> Highest amount of Saturation:</p> (' + indexMaxValueS + ', ' + histogramS[indexMaxValueS] + ')</div>';
    document.getElementById('highestValues').innerHTML += '<div><p class="fw-bold d-inline"> Highest amount of Value:</p> (' + indexMaxValueV + ', ' + histogramV[indexMaxValueV] + ')</div>';
}

// Function to create the histogram chart
function createHistogramChartGrayScale(histogramData) {
    initChartDiv();

    // Define labels for histogram bins (intensity levels)
    var labels = Array.from(Array(256).keys());

    // Create the dataset
    var dataset = {
        label: 'Gray Scale',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        borderColor: 'rgb(255, 99, 132)',
        data: histogramData
    };

    // Configuration options for the chart
    var options = {
        scales: {
            y: {
                beginAtZero: true
            }
        }
    };

    // Create the data object
    var data = {
        labels: labels,
        datasets: [dataset]
    };

    // Get the canvas element
    var ctx = document.getElementById('myChart').getContext('2d');

    // Create the chart
    var histogramChart = new Chart(ctx, {
        type: 'bar',
        data: data,
        options: options
    });
    // Get the text element #highestValue and change its text to "Highest Value: (" index + ',' + highestValue + ')'
    let indexMaxValue = histogramData.indexOf(Math.max(...histogramData));
    document.getElementById('highestValues').innerHTML = '<p class="fw-bold d-inline"> Highest Value:</p> (' + indexMaxValue + ', ' + histogramData[indexMaxValue] + ')';
}