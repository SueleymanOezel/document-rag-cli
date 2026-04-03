# document-rag-cli

Ein einfaches RAG-CLI-Tool fuer Fragen zu Textdokumenten mit Gemini.

## API-Key einrichten

Dieses Projekt liest den Gemini-Key aus der Umgebungsvariable `GEMINI_API_KEY`.

### Empfohlen: `.env` (kein manuelles Setzen pro Terminal)

1. Erstelle im Projektordner eine Datei `.env`
2. Trage deinen Key ein:

```env
GEMINI_API_KEY=DEIN_NEUER_KEY
```

Beim Start wird `.env` automatisch geladen.

### PowerShell (nur fuer aktuelle Session)

```powershell
$env:GEMINI_API_KEY="DEIN_NEUER_KEY"
```

Wichtig: kein Backslash vor `$env`.

### PowerShell dauerhaft (fuer neue Terminals)

```powershell
setx GEMINI_API_KEY "DEIN_NEUER_KEY"
```

Danach ein neues Terminal oeffnen (oder VS Code neu starten), damit der Wert verfuegbar ist.