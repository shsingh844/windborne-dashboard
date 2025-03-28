// Utility functions for the Windborne Constellation Tracker

/**
 * Format a timestamp to a human-readable date/time
 * @param {number} timestamp - Unix timestamp in seconds
 * @returns {string} Formatted date/time
 */
function formatTimestamp(timestamp) {
    if (!timestamp) return 'Unknown';
    
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
}

/**
 * Calculate the great-circle distance between two points
 * @param {number} lat1 - Latitude of first point in degrees
 * @param {number} lon1 - Longitude of first point in degrees
 * @param {number} lat2 - Latitude of second point in degrees
 * @param {number} lon2 - Longitude of second point in degrees
 * @returns {number} Distance in kilometers
 */
function calculateDistance(lat1, lon1, lat2, lon2) {
    // Earth radius in kilometers
    const earthRadius = 6371;
    
    // Convert to radians
    const lat1Rad = toRadians(lat1);
    const lat2Rad = toRadians(lat2);
    const lonDiff = toRadians(lon2 - lon1);
    
    // Haversine formula
    const a = Math.sin((lat2Rad - lat1Rad) / 2) * Math.sin((lat2Rad - lat1Rad) / 2) +
              Math.cos(lat1Rad) * Math.cos(lat2Rad) * 
              Math.sin(lonDiff / 2) * Math.sin(lonDiff / 2);
    
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    const distance = earthRadius * c;
    
    return distance;
}

/**
 * Convert degrees to radians
 * @param {number} degrees - Angle in degrees
 * @returns {number} Angle in radians
 */
function toRadians(degrees) {
    return degrees * (Math.PI / 180);
}

/**
 * Calculate bearing between two points
 * @param {number} lat1 - Latitude of first point in degrees
 * @param {number} lon1 - Longitude of first point in degrees
 * @param {number} lat2 - Latitude of second point in degrees
 * @param {number} lon2 - Longitude of second point in degrees
 * @returns {number} Bearing in degrees (0-360)
 */
function calculateBearing(lat1, lon1, lat2, lon2) {
    const lat1Rad = toRadians(lat1);
    const lat2Rad = toRadians(lat2);
    const lonDiff = toRadians(lon2 - lon1);
    
    const y = Math.sin(lonDiff) * Math.cos(lat2Rad);
    const x = Math.cos(lat1Rad) * Math.sin(lat2Rad) -
              Math.sin(lat1Rad) * Math.cos(lat2Rad) * Math.cos(lonDiff);
    
    let bearing = Math.atan2(y, x);
    bearing = toDegrees(bearing);
    bearing = (bearing + 360) % 360;
    
    return bearing;
}

/**
 * Convert radians to degrees
 * @param {number} radians - Angle in radians
 * @returns {number} Angle in degrees
 */
function toDegrees(radians) {
    return radians * (180 / Math.PI);
}

/**
 * Get cardinal direction from bearing
 * @param {number} bearing - Bearing in degrees
 * @returns {string} Cardinal direction (N, NE, E, etc.)
 */
function getCardinalDirection(bearing) {
    const directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    const index = Math.round(bearing / 45) % 8;
    return directions[index];
}

/**
 * Format a number with commas as thousands separators
 * @param {number} num - The number to format
 * @returns {string} Formatted number
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Get a color based on a value within a range
 * @param {number} value - The value to map to a color
 * @param {number} min - Minimum value of the range
 * @param {number} max - Maximum value of the range
 * @returns {string} RGB color string
 */
function getColorForValue(value, min, max) {
    // Normalize value to 0-1 range
    const normalized = Math.max(0, Math.min(1, (value - min) / (max - min)));
    
    // Generate color: blue (low) to red (high)
    const r = Math.floor(normalized * 255);
    const g = Math.floor((1 - Math.abs(normalized - 0.5) * 2) * 255);
    const b = Math.floor((1 - normalized) * 255);
    
    return `rgb(${r},${g},${b})`;
}

/**
 * Check if a balloon is active based on timestamp
 * @param {number} timestamp - Balloon's last update timestamp
 * @param {number} threshold - Threshold in seconds (default: 2 hours)
 * @returns {boolean} True if balloon is active
 */
function isBalloonActive(timestamp, threshold = 7200) {
    if (!timestamp) return false;
    return (Date.now() / 1000 - timestamp) < threshold;
}

/**
 * Debounce function to limit how often a function can be called
 * @param {Function} func - The function to debounce
 * @param {number} wait - The debounce wait time in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        const context = this;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), wait);
    };
}