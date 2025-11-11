# Rolling Window vs Expanding Window - Hangisi Gerekli?

## 📊 Karşılaştırma

### Expanding Window (Şu An Mevcut)

```
Fold 1: Train [0-10k] → Test [10k-12k]
Fold 2: Train [0-12k] → Test [12k-14k]  ← Tüm geçmiş veriler
Fold 3: Train [0-14k] → Test [14k-16k]  ← Tüm geçmiş veriler
Fold 4: Train [0-16k] → Test [16k-18k]  ← Tüm geçmiş veriler
Fold 5: Train [0-18k] → Test [18k-20k]  ← Tüm geçmiş veriler

✅ Avantajlar:
- Tüm geçmiş verileri kullanır
- Uzun vadeli trendleri yakalar
- Daha fazla veri = daha stabil model

⚠️ Dezavantajlar:
- Eski veriler gereksiz olabilir (2020 verisi 2024 için çok eski)
- Trend değişikliklerine yavaş adapte olur
- Büyük veri setlerinde hesaplama maliyeti yüksek
```

### Rolling Window (Şu An Yok)

```
Fold 1: Train [0-10k] → Test [10k-12k]
Fold 2: Train [2k-12k] → Test [12k-14k]  ← İlk 2k çıkarıldı
Fold 3: Train [4k-14k] → Test [14k-16k]  ← İlk 4k çıkarıldı
Fold 4: Train [6k-16k] → Test [16k-18k]  ← İlk 6k çıkarıldı
Fold 5: Train [8k-18k] → Test [18k-20k]  ← İlk 8k çıkarıldı

✅ Avantajlar:
- Sadece son verileri kullanır (daha güncel)
- Trend değişikliklerine hızlı adapte olur
- Market regime değişikliklerine daha iyi uyum
- Büyük veri setlerinde daha hızlı

⚠️ Dezavantajlar:
- Eski verileri kaybeder
- Uzun vadeli pattern'leri kaçırabilir
- Daha az veri = daha az stabil
```

## 🎯 Kripto Verisi için Hangisi?

### Expanding Window Kullan:
- ✅ Uzun vadeli trend analizi yapıyorsanız
- ✅ Tüm geçmiş pattern'leri öğrenmek istiyorsanız
- ✅ Küçük-orta veri setleri (< 20,000 satır)
- ✅ Model'in tüm geçmişten öğrenmesini istiyorsanız

### Rolling Window Kullan:
- ✅ **Büyük veri setleri (> 50,000 satır)** ⭐
- ✅ **Market regime değişiklikleri sık oluyorsa** ⭐
- ✅ **Eski veriler gereksiz olabilir** ⭐
- ✅ **Trend değişikliklerine hızlı adapte olmak istiyorsanız** ⭐
- ✅ **Model drift riski yüksekse** ⭐

## 💡 Büyük Veri Setleri için Örnek

### Senaryo: 50,000 satır Bitcoin verisi

#### Expanding Window:
```
Fold 5: Train [0-40k] → Test [40k-42k]
  ↓
Problem: İlk 10,000 satır (2020) çok eski!
- 2020 pattern'leri 2024 için geçerli mi?
- Model eski verilerle "kirletilmiş" olabilir
- Hesaplama maliyeti yüksek (40k satır)
```

#### Rolling Window (10k sabit pencere):
```
Fold 5: Train [30k-40k] → Test [40k-42k]
  ↓
Avantaj: Sadece son 10,000 satır (2023-2024)
- Daha güncel ve relevant veriler
- Trend değişikliklerine hızlı adapte
- Hesaplama maliyeti düşük (10k satır)
- Market regime değişikliklerine daha iyi uyum
```

## 🚀 Öneri: Her İkisini de Destekle

### En İyi Yaklaşım:

```python
# Config'de seçim yapılabilir
walk_forward_type: str = "expanding"  # veya "rolling"
walk_forward_window_size: Optional[int] = None  # Rolling için pencere boyutu
```

### Kullanım Senaryoları:

#### Senaryo 1: Küçük Veri Seti (< 20k satır)
```python
# Expanding window kullan
config.model.walk_forward_type = "expanding"
```

#### Senaryo 2: Büyük Veri Seti (> 50k satır)
```python
# Rolling window kullan
config.model.walk_forward_type = "rolling"
config.model.walk_forward_window_size = 10000  # Son 10k satır
```

#### Senaryo 3: Her İkisini de Test Et
```python
# Önce expanding ile test et
results_expanding = trainer.walk_forward_validation(type="expanding")

# Sonra rolling ile test et
results_rolling = trainer.walk_forward_validation(type="rolling", window_size=10000)

# Karşılaştır ve en iyisini seç
```

## ✅ Sonuç

### Rolling Window Gerekli mi?

**Büyük veri setleri için: EVET, önerilir!**

Nedenler:
1. ✅ **Büyük veri setleri**: Eski veriler gereksiz olabilir
2. ✅ **Kripto verisi**: Market regime değişiklikleri sık
3. ✅ **Model drift**: Trend değişikliklerine hızlı adapte olur
4. ✅ **Hesaplama**: Daha hızlı (daha az veri)
5. ✅ **Güncellik**: Sadece son verileri kullanır

### Öneri:

**Her ikisini de destekle** - Kullanıcı seçsin veya her ikisini de test etsin!

