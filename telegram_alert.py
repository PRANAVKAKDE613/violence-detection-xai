import asyncio
import telegram

# Your credentials
BOT_TOKEN = "8371384587:AAFk-u3PBqKhcZuEFppvxuTAGCnSV6CAD1s"
CHAT_ID = "1094163219"

async def send_alert(image_path, label, confidence):
    bot = telegram.Bot(token=BOT_TOKEN)
    
    # Send text message
    message = (
        f"🚨 *VIOLENCE DETECTION ALERT* 🚨\n\n"
        f"Status: *{label}*\n"
        f"Confidence: *{confidence:.2%}*\n"
        f"System is monitoring..."
    )
    await bot.send_message(
        chat_id=CHAT_ID,
        text=message,
        parse_mode='Markdown'
    )
    
    # Send Grad-CAM image
    with open(image_path, 'rb') as img:
        await bot.send_photo(
            chat_id=CHAT_ID,
            photo=img,
            caption=f"Grad-CAM Heatmap — {label} ({confidence:.2%})"
        )
    
    print("Alert sent to Telegram!")

if __name__ == "__main__":
    asyncio.run(send_alert(
        image_path='gradcam_output.png',
        label='VIOLENCE',
        confidence=0.6803
    ))