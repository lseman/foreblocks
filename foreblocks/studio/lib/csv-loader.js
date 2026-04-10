function parseCsvLine(line, delimiter) {
  const cells = [];
  let current = "";
  let inQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        current += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === delimiter && !inQuotes) {
      cells.push(current);
      current = "";
      continue;
    }

    current += char;
  }

  cells.push(current);
  return cells.map((cell) => cell.trim());
}

function detectDelimiter(headerLine) {
  const commas = (headerLine.match(/,/g) || []).length;
  const semicolons = (headerLine.match(/;/g) || []).length;
  return semicolons > commas ? ";" : ",";
}

function normalizeHeader(value) {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function isFiniteNumber(value) {
  if (value == null) {
    return false;
  }

  const trimmed = String(value).trim();
  if (!trimmed) {
    return false;
  }

  return Number.isFinite(Number(trimmed));
}

function isLikelyDatetime(value) {
  if (value == null) {
    return false;
  }

  const trimmed = String(value).trim();
  if (!trimmed) {
    return false;
  }

  const hasDateShape =
    /[-/:T]/.test(trimmed) ||
    /jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec/i.test(trimmed) ||
    /^\d{4}(?:\d{2}){1,2}$/.test(trimmed);

  if (!hasDateShape) {
    return false;
  }

  return !Number.isNaN(Date.parse(trimmed));
}

function inspectColumns(headers, rows) {
  return headers.map((header, index) => {
    let missingCount = 0;
    let numericCount = 0;
    let dateLikeCount = 0;
    let sampleValue = "";

    for (const cells of rows) {
      const value = (cells[index] ?? "").trim();

      if (!value) {
        missingCount += 1;
        continue;
      }

      if (!sampleValue) {
        sampleValue = value;
      }

      if (isFiniteNumber(value)) {
        numericCount += 1;
      }

      if (isLikelyDatetime(value)) {
        dateLikeCount += 1;
      }
    }

    const nonEmptyCount = rows.length - missingCount;

    return {
      name: header,
      missingCount,
      nonEmptyCount,
      numericCount,
      dateLikeCount,
      sampleValue,
    };
  });
}

function scoreTimestampHeader(header, stats) {
  const normalized = normalizeHeader(header);
  let score = 0;

  if (["date", "time", "timestamp", "datetime", "ds"].includes(normalized)) {
    score += 12;
  }

  if (/date|time|timestamp|datetime|period|month|day|hour|week|year/.test(normalized)) {
    score += 8;
  }

  const dateRatio = stats.nonEmptyCount > 0 ? stats.dateLikeCount / stats.nonEmptyCount : 0;
  score += dateRatio * 10;

  return score;
}

function scoreTargetHeader(header, stats) {
  const normalized = normalizeHeader(header);
  let score = 0;

  if (/date|time|timestamp|datetime|period|month|day|hour|week|year/.test(normalized)) {
    score -= 10;
  }

  if (["target", "y", "value", "signal"].includes(normalized)) {
    score += 12;
  }

  if (/target|value|signal|load|demand|sales|price|energy|power|flow|inflow|outflow|volume|level|temp|rain/.test(normalized)) {
    score += 8;
  }

  const numericRatio = stats.nonEmptyCount > 0 ? stats.numericCount / stats.nonEmptyCount : 0;
  score += numericRatio * 10;

  return score;
}

function suggestColumns(headers, columnStats) {
  const rankedTimestamp = headers
    .map((header, index) => ({
      header,
      score: scoreTimestampHeader(header, columnStats[index]),
      stats: columnStats[index],
    }))
    .filter((item) => item.score > 0 && item.stats.nonEmptyCount > 0)
    .sort((left, right) => right.score - left.score);

  const timeColumn = rankedTimestamp[0]?.header ?? "";

  const rankedTarget = headers
    .map((header, index) => ({
      header,
      score: scoreTargetHeader(header, columnStats[index]),
      stats: columnStats[index],
    }))
    .filter((item) => item.header !== timeColumn)
    .filter((item) => item.stats.numericCount > 0)
    .sort((left, right) => right.score - left.score);

  return {
    targetColumn: rankedTarget[0]?.header ?? "",
    timeColumn,
  };
}

export function inspectCsvText(csvText) {
  const trimmed = csvText.trim();
  if (!trimmed) {
    throw new Error("CSV file is empty.");
  }

  const lines = trimmed.split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) {
    throw new Error("CSV file must contain a header and at least one data row.");
  }

  const delimiter = detectDelimiter(lines[0]);
  const headers = parseCsvLine(lines[0], delimiter);
  const rows = lines.slice(1).map((line) => parseCsvLine(line, delimiter));
  const columnStats = inspectColumns(headers, rows);
  const suggestions = suggestColumns(headers, columnStats);
  const missingCellCount = rows.reduce(
    (total, cells) =>
      total + headers.reduce((rowTotal, _header, index) => rowTotal + ((cells[index] ?? "").trim() ? 0 : 1), 0),
    0,
  );

  return {
    delimiter,
    headers,
    rows,
    columnStats,
    suggestions,
    summary: {
      rowCount: rows.length,
      columnCount: headers.length,
      missingCellCount,
    },
  };
}

export function parseSeriesCsvText(csvText, { targetColumn, timeColumn }) {
  const inspection = inspectCsvText(csvText);
  const activeTargetColumn = targetColumn || inspection.suggestions.targetColumn;
  const activeTimeColumn = timeColumn || inspection.suggestions.timeColumn;
  const targetIndex = inspection.headers.indexOf(activeTargetColumn);
  const timeIndex = inspection.headers.indexOf(activeTimeColumn);
  const covariateColumns = inspection.headers.filter(
    (header) => header !== activeTargetColumn && header !== activeTimeColumn,
  );

  if (targetIndex === -1) {
    throw new Error(`Target column \"${activeTargetColumn}\" was not found in the CSV header.`);
  }

  if (activeTimeColumn && timeIndex === -1) {
    throw new Error(`Timestamp column \"${activeTimeColumn}\" was not found in the CSV header.`);
  }

  let missingTargetCount = 0;
  let invalidTargetCount = 0;
  let missingTimestampCount = 0;

  const records = inspection.rows
    .map((cells, index) => {
      const rawValue = (cells[targetIndex] ?? "").trim();

      if (!rawValue) {
        missingTargetCount += 1;
        return null;
      }

      const value = Number(rawValue);

      if (!Number.isFinite(value)) {
        invalidTargetCount += 1;
        return null;
      }

      const rawTimestamp = timeIndex >= 0 ? (cells[timeIndex] ?? "").trim() : "";
      if (timeIndex >= 0 && !rawTimestamp) {
        missingTimestampCount += 1;
      }

      const timestamp = rawTimestamp || String(index + 1);
      const parsedTime = Date.parse(timestamp);
      const covariates = Object.fromEntries(
        covariateColumns.map((header) => {
          const columnIndex = inspection.headers.indexOf(header);
          const rawCovariate = (cells[columnIndex] ?? "").trim();
          const numericValue = Number(rawCovariate);

          return [
            header,
            rawCovariate && Number.isFinite(numericValue) ? numericValue : null,
          ];
        }),
      );

      return {
        value,
        timestamp,
        sortKey: Number.isNaN(parsedTime) ? index : parsedTime,
        covariates,
      };
    })
    .filter(Boolean)
    .sort((left, right) => left.sortKey - right.sortKey);

  if (records.length < 24) {
    throw new Error("At least 24 numeric observations are required to compute ACF/PACF diagnostics.");
  }

  const series = records.map((point, index) => ({
    t: index,
    value: point.value,
    timestamp: point.timestamp,
  }));

  const covariates = covariateColumns
    .map((header) => {
      const values = records.map((point) => point.covariates[header]);
      const missingCount = values.filter((value) => value == null).length;
      const validCount = values.length - missingCount;

      return {
        name: header,
        missingCount,
        validCount,
        validRatio: values.length > 0 ? validCount / values.length : 0,
        values,
        points: values.map((value, index) => ({
          t: index,
          value,
          timestamp: records[index].timestamp,
        })),
      };
    })
    .filter((column) => column.validCount >= 12);

  return {
    headers: inspection.headers,
    suggestions: inspection.suggestions,
    summary: {
      ...inspection.summary,
      targetColumn: activeTargetColumn,
      timeColumn: activeTimeColumn,
      validObservationCount: records.length,
      missingTargetCount,
      invalidTargetCount,
      missingTimestampCount,
      droppedRowCount: inspection.summary.rowCount - records.length,
      exogenousCount: covariates.length,
    },
    series,
    covariates,
  };
}

export function extractSeriesFromCsvText(csvText, { targetColumn, timeColumn }) {
  return parseSeriesCsvText(csvText, { targetColumn, timeColumn }).series;
}
