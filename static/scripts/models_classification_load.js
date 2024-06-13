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

$(document).ready(function () {
    function generateBinaryTable(data) {
        html = `
        <table class="table table-bordered table-striped table-hover table-sm">
                <thead>
                    <tr>
                        <th scope="col">Property/Class</th>
                        <th scope="col">Negative</th>
                        <th scope="col">Positive</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <th scope="row">Precision</th>
                        <td>${parseFloat(data.negative.precision.toFixed(2))}</td>
                        <td>${parseFloat(data.positive.precision.toFixed(2))}</td>
                    </tr>
                    <tr>
                        <th scope="row">Recall</th>
                        <td>${parseFloat(data.negative.recall.toFixed(2))}</td>
                        <td>${parseFloat(data.positive.recall.toFixed(2))}</td>
                    </tr>
                    <tr>
                        <th scope="row">F1-score</th>
                        <td>${parseFloat(data.negative.f1_score.toFixed(2))}</td>
                        <td>${parseFloat(data.positive.f1_score.toFixed(2))}</td>
                    </tr>
                    <tr>
                        <th scope="row">Support</th>
                        <td>${parseFloat(data.negative.support.toFixed(2))}</td>
                        <td>${parseFloat(data.positive.support.toFixed(2))}</td>
                    </tr>
                </tbody>
                
            </table>
        `;
        return html;
    }

    function generateMulticlassTable(data) {
        html = `
        <table class="table table-bordered table-striped table-hover table-sm">
                <thead>
                    <tr>
                        <th scope="col">Property/Class</th>
                        <th scope="col">H</th>
                        <th scope="col">US</th>
                        <th scope="col">HSIL</th>
                        <th scope="col">LSIL</th>
                        <th scope="col">Negative</th>
                        <th scope="col">SCC</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <th scope="row">Precision</th>
                        <td>${parseFloat(data.ASC_H.precision.toFixed(2))}</td>
                        <td>${parseFloat(data.ASC_US.precision.toFixed(2))}</td>
                        <td>${parseFloat(data.HSIL.precision.toFixed(2))}</td>
                        <td>${parseFloat(data.LSIL.precision.toFixed(2))}</td>
                        <td>${parseFloat(data.negative.precision.toFixed(2))}</td>
                        <td>${parseFloat(data.SCC.precision.toFixed(2))}</td>
                    </tr>
                    <tr>
                        <th scope="row">Recall</th>
                        <td>${parseFloat(data.ASC_H.recall.toFixed(2))}</td>
                        <td>${parseFloat(data.ASC_US.recall.toFixed(2))}</td>
                        <td>${parseFloat(data.HSIL.recall.toFixed(2))}</td>
                        <td>${parseFloat(data.LSIL.recall.toFixed(2))}</td>
                        <td>${parseFloat(data.negative.recall.toFixed(2))}</td>
                        <td>${parseFloat(data.SCC.recall.toFixed(2))}</td>
                    </tr>
                    <tr>
                        <th scope="row">F1-score</th>
                        <td>${parseFloat(data.ASC_H.f1_score.toFixed(2))}</td>
                        <td>${parseFloat(data.ASC_US.f1_score.toFixed(2))}</td>
                        <td>${parseFloat(data.HSIL.f1_score.toFixed(2))}</td>
                        <td>${parseFloat(data.LSIL.f1_score.toFixed(2))}</td>
                        <td>${parseFloat(data.negative.f1_score.toFixed(2))}</td>
                        <td>${parseFloat(data.SCC.f1_score.toFixed(2))}</td>
                    </tr>
                    <tr>
                        <th scope="row">Support</th>
                        <td>${parseFloat(data.ASC_H.support.toFixed(2))}</td>
                        <td>${parseFloat(data.ASC_US.support.toFixed(2))}</td>
                        <td>${parseFloat(data.HSIL.support.toFixed(2))}</td>
                        <td>${parseFloat(data.LSIL.support.toFixed(2))}</td>
                        <td>${parseFloat(data.negative.support.toFixed(2))}</td>
                        <td>${parseFloat(data.SCC.support.toFixed(2))}</td>
                    </tr>
                </tbody>
                
            </table>
        `;
        return html;
    }

    async function getModelData(model) {
        var csrftoken = getCookie('csrftoken');
        try {
            const response = await fetch(modelsClassification + '?model=' + model, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrftoken
                },
            });
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error:', error);
            return null;
        }
    }

    document.querySelectorAll('.xgb-selector').forEach(radio => {
        radio.addEventListener('change', async function () {
            if (this.checked) {
                const selectedButton = this.id.replace('vbtn-', '');
                if (selectedButton === 'xgboost-binary') {
                    data = await getModelData('xgboostBinary');
                    if (data) {
                        document.getElementById('xgb-values').innerHTML = generateBinaryTable(data);
                    }
                } else if (selectedButton === 'xgboost-multiclass') {
                    data = await getModelData('xgboostMulticlass');
                    if (data) {
                        document.getElementById('xgb-values').innerHTML = generateMulticlassTable(data);
                    }
                }
            }
        });
    });

    document.querySelectorAll('.efficient-selector').forEach(radio => {
        radio.addEventListener('change', async function () {
            if (this.checked) {
                const selectedButton = this.id.replace('vbtn-', '');
                if (selectedButton === 'efficientnet-binary') {
                    data = await getModelData('efficientNetBinary');
                    if (data) {
                        document.getElementById('efficientnet-values').innerHTML = generateBinaryTable(data);
                    }
                } else if (selectedButton === 'efficientnet-multiclass') {
                    data = await getModelData('efficientNetMulticlass');
                    if (data) {
                        document.getElementById('efficientnet-values').innerHTML = generateMulticlassTable(data);
                    }
                }
            }
        });
    });

    var csrftoken = getCookie('csrftoken');
    fetch(modelsClassification + '?model=xgboostBinary', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrftoken
        },
    })
        .then(response => response.json())
        .then(data => {
            html = generateBinaryTable(data);
            $('#xgb-values').html(html);
        })
        .catch(error => {
        });

    fetch(modelsClassification + '?model=efficientNetBinary', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrftoken
        },
    })
        .then(response => response.json())
        .then(data => {
            html = generateBinaryTable(data);
            $('#efficientnet-values').html(html);
        })
        .catch(error => {
            console.error('Error:', error);
        });
});