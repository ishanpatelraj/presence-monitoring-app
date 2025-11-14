async function loadUsers() {
    const res = await fetch("/users");
    const data = await res.json();

    const table = document.querySelector("#users-table tbody");
    table.innerHTML = "";

    data.forEach(user => {
        const row = `
            <tr>
                <td>${user.name}</td>
                <td>${user.user_id}</td>
                <td>${user.created_at}</td>
            </tr>
        `;
        table.innerHTML += row;
    });
}

async function loadLogs() {
    const res = await fetch("/logs");
    const data = await res.json();

    const table = document.querySelector("#logs-table tbody");
    table.innerHTML = "";

    data.forEach(log => {
        const row = `
            <tr>
                <td>${log.time}</td>
                <td>${log.user}</td>
                <td>${log.status}</td>
                <td>${log.alerts}</td>
            </tr>
        `;
        table.innerHTML += row;
    });
}

setInterval(() => {
    loadUsers();
    loadLogs();
}, 2000);
