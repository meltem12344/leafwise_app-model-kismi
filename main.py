# main.py

"""
API Gateway / Orchestration Service

Akış:
1. Mobile App → Fotoğraf yükler
2. Vision Service → Hastalık tahmin eder
3. Retrieval Service → RAG'den bilgi çeker
4. Generation Service → Tavsiye oluşturur
5. Response → Kullanıcıya döner
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from PIL import Image
import io
import time
from datetime import datetime

# Servisler
from storage import storage
from services.vision_service import vision_service
from services.retrieval_service import retrieval_service
from services.generation_service import generation_service
from services.location_service import get_city_from_coordinates
from services.weather_service import get_extended_weather
from services.logging_service import logger

app = FastAPI(title="Tarim AI Backend - RAG Architecture")

@app.get("/")
def home():
    """Sistem durumu"""
    return {
        "message": "Tarim AI Backend",
        "status": "running",
        "architecture": "RAG-based",
        "services": {
            "vision": "MobileNet (fine-tuned)",
            "retrieval": retrieval_service.get_stats(),
            "generation": "Gemini 1.5 Flash",
            "storage": storage.__class__.__name__
        }
    }

@app.post("/api/v1/analyze")
async def analyze(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    plant: str = Form(default="Domates")
):
    """
    Fotoğraf analizi - RAG Pipeline
    
    Akış:
    1. Vision: Hastalık tespiti
    2. Location: GPS → Şehir
    3. Weather: Hava durumu + tahmin
    4. Retrieval: RAG'den bilgi çek (metadata filtering)
    5. Generation: Kaynak-temelli tavsiye oluştur
    6. Storage: Düşük confidence → fotoğraf kaydet
    7. Response: Kullanıcıya döndür
    """
    
    start_time = time.time()
    
    # Log: İstek
    logger.log_request("/api/v1/analyze", {
        "latitude": latitude,
        "longitude": longitude,
        "plant": plant,
        "filename": file.filename
    })
    
    print("\n" + "="*60)
    print(" RAG PIPELINE BAŞLIYOR")
    print("="*60)
    
    try:
        # 1. DOSYA KONTROLÜ
        if not file.content_type.startswith("image/"):
            logger.log_error("Geçersiz dosya tipi", {
                "content_type": file.content_type
            })
            raise HTTPException(400, "Sadece fotoğraf kabul edilir!")
        
        # 2. RAM'DE OKU
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        original_size = image.size
        size_kb = len(contents) / 1024
        
        print(f" Dosya: {file.filename}")
        print(f" Boyut: {original_size}, {size_kb:.0f} KB")
        print(f" Konum: ({latitude}, {longitude})")
        
        # 3. VISION SERVICE - HASTALIK TESPİTİ
        print("\n Vision Service - Hastalık tespiti...")
        
        vision_result = vision_service.predict(contents)
        disease = vision_result['disease']
        confidence = vision_result['confidence']
        
        print(f"    Tespit: {disease} (%{confidence*100:.1f})")
        print(f"    Top-3: {vision_result['all_predictions'][:3]}")
        
        # 4. LOCATION SERVICE - GPS → ŞEHİR
        print("\n Location Service...")
        city_info = get_city_from_coordinates(latitude, longitude)
        
        if not city_info:
            logger.log_error("Konum bulunamadı", {
                "latitude": latitude,
                "longitude": longitude
            })
            raise HTTPException(400, "Konum bulunamadı")
        
        city = city_info['city']
        print(f"    Şehir: {city}")
        
        # 5. WEATHER SERVICE - HAVA DURUMU + TAHMİN
        print("\n  Weather Service...")
        extended_weather = get_extended_weather(city)
        
        if extended_weather:
            current = extended_weather['current']
            forecast = extended_weather['forecast']
            spray_advice = extended_weather['advice']
            
            print(f"    Şu an: {current['temperature']}°C, {current['description']}")
            print(f"    24 saat: {forecast['min_temp']:.1f}°C - {forecast['max_temp']:.1f}°C")
            print(f"   {'✅' if spray_advice['can_spray'] else '❌'} İlaçlama: {spray_advice['reason']}")
        else:
            print("     Hava durumu alınamadı")
            extended_weather = None
        
        # 6. RETRIEVAL SERVICE - RAG RETRIEVAL
        print("\n🔍 Retrieval Service - RAG...")
        
        rag_results = retrieval_service.retrieve(
            disease=disease,
            plant=plant,
            top_k=5
        )
        
        if rag_results:
            print(f"    {len(rag_results)} döküman bulundu")
            for i, result in enumerate(rag_results[:3], 1):
                print(f"   📄 Döküman {i}: Score {result['score']:.2f} - {result['source']}")
        else:
            print("     RAG'de döküman bulunamadı")
        
        # 7. GENERATION SERVICE - LLM TAVSIYE
        print("\n Generation Service - LLM...")
        
        advice = generation_service.generate(
            disease=disease,
            confidence=confidence,
            plant=plant,
            city=city,
            weather=extended_weather,
            rag_results=rag_results
        )
        
        print("    Tavsiye oluşturuldu")
        
        # 8. STORAGE - DÜŞÜK CONFIDENCE
        photo_path = None
        confidence_threshold = 0.80
        
        if confidence < confidence_threshold:
            print(f"\n  Düşük confidence ({confidence:.2f} < {confidence_threshold})")
            print("   Fotoğraf kaydediliyor...")
            
            # Küçült
            small = image.resize((224, 224), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            small.save(buffer, format='JPEG', quality=70, optimize=True)
            small_bytes = buffer.getvalue()
            
            # Kaydet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{timestamp}.jpg"
            
            photo_path = storage.save_photo(
                image_data=small_bytes,
                filename=filename,
                metadata={
                    'city': city,
                    'disease': disease,
                    'confidence': confidence,
                    'plant': plant
                }
            )
            print(f"    Kaydedildi: {photo_path}")
        else:
            print(f"\n Yüksek confidence ({confidence:.2f}), sadece metadata")
        
        # 9. LOGGİNG
        duration = time.time() - start_time
        
        logger.log_analysis({
            "timestamp": datetime.now().isoformat(),
            "city": city,
            "disease": disease,
            "confidence": confidence,
            "plant": plant,
            "rag_results_count": len(rag_results),
            "photo_saved": photo_path is not None,
            "processing_time": round(duration, 2)
        })
        
        logger.log_response("/api/v1/analyze", 200, duration)
        
        print("\n RAG PIPELINE TAMAMLANDI")
        print(f"⏱  Toplam süre: {duration:.2f}s")
        print("="*60 + "\n")
        
        # 10. RESPONSE
        return {
            "success": True,
            "vision": {
                "disease": disease,
                "confidence": round(confidence, 3),
                "all_predictions": vision_result['all_predictions']
            },
            "location": {
                "city": city,
                "latitude": latitude,
                "longitude": longitude
            },
            "weather": {
                "current": extended_weather['current'] if extended_weather else None,
                "forecast": extended_weather['forecast'] if extended_weather else None,
                "spray_advice": extended_weather['advice'] if extended_weather else None
            },
            "retrieval": {
                "results_found": len(rag_results),
                "top_sources": [r['source'] for r in rag_results[:3]],
                "scores": [round(r['score'], 3) for r in rag_results[:3]]
            },
            "plant": plant,
            "advice": advice,
            "photo_saved": photo_path is not None,
            "photo_path": photo_path,
            "processing_time": round(duration, 2)
        }
        
    except HTTPException as http_exc:
        duration = time.time() - start_time
        logger.log_response("/api/v1/analyze", http_exc.status_code, duration)
        raise
        
    except Exception as e:
        print(f" HATA: {e}")
        import traceback
        traceback.print_exc()
        
        duration = time.time() - start_time
        
        logger.log_error(f"Pipeline hatası: {str(e)}", {
            "latitude": latitude,
            "longitude": longitude,
            "plant": plant,
            "traceback": traceback.format_exc()
        })
        
        logger.log_response("/api/v1/analyze", 500, duration)
        raise HTTPException(500, str(e))

@app.get("/api/v1/stats")
def get_stats():
    """Sistem istatistikleri"""
    return {
        "retrieval": retrieval_service.get_stats(),
        "vision": {
            "model": "MobileNet-v2 (fine-tuned)",
            "classes": len(vision_service.class_names)
        },
        "generation": {
            "model": "Gemini 1.5 Flash",
            "enabled": generation_service.enabled
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("\n Tarim AI Backend başlatılıyor...")
    print(" Mimari: RAG-based")
    print(f" Storage: {storage.__class__.__name__}")
    print(f" Retrieval: {retrieval_service.get_stats()}")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)