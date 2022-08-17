// Create WebSocket connection.
window.onload = ()=>{
    const socket = new WebSocket('ws://127.0.0.1:8080');
    var canvas = document.getElementById("c");
    var ctx = canvas.getContext("2d");
    var image = new Image();
    image.onload = function() {
        ctx.drawImage(image, 0, 0);
    };

// Connection opened
    socket.addEventListener('open', function (event) {
        socket.send('Hello Server!');
    });

// Listen for messages
    socket.addEventListener('message', function (event) {
        const data = event.data
        image.src = data;
    });
}
