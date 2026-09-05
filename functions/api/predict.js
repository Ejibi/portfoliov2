
// functions/api/predict.js
export async function onRequestPost(context) {
  try {
    const request = context.request;
    const authHeader = request.headers.get("Authorization");

    // (Optional) Validate Firebase Auth JWT token here if needed
    if (!authHeader) {
      return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401 });
    }

    // Forward file payload directly to Modal Webhook
    const formData = await request.formData();
    const modalUrl = context.env.MODAL_WEBHOOK_URL; // Set in Cloudflare Dashboard

    const modalResponse = await fetch(modalUrl, {
      method: "POST",
      body: formData,
    });

    const data = await modalResponse.json();

    return new Response(JSON.stringify(data), {
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: err.message }), { status: 500 });
  }
}